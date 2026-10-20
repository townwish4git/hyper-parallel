# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Opt-in pointwise Muon operations with explicit backend requirements."""
from functools import lru_cache

import torch  # pylint: disable=forbidden-backend-import


@lru_cache(maxsize=1)
def _npu_kernels():
    # Triton is optional and must not be imported by ordinary optimizer users.
    try:
        from hyper_parallel.core.optimizer import _muon_triton  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("Muon triton paths require triton-ascend in the active NPU environment") from exc
    return _muon_triton


def _validate_tensors(tensors):
    for tensor in tensors:
        if tensor.device.type != "npu" or not tensor.is_contiguous():
            raise ValueError("Muon triton paths require contiguous local NPU tensors")
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Muon triton paths require float32, float16 or bfloat16 tensors")


def _momentum_update_triton(grads, bufs, momentum1, momentum2, nesterov):
    """Return independent BF16 directions while preserving buffer state semantics."""
    if len(grads) != len(bufs):
        raise ValueError("Muon gradient and buffer counts must match")
    _validate_tensors([*grads, *bufs])
    for grad, buf in zip(grads, bufs):
        if grad.shape != buf.shape or grad.dtype != buf.dtype or grad.device != buf.device:
            raise ValueError("Muon gradients and buffers must have matching shape, dtype and device")
    kernel = _npu_kernels()._momentum_kernel  # pylint: disable=protected-access
    outputs = [torch.empty_like(grad, dtype=torch.bfloat16) for grad in grads]
    buckets = {}
    for index, grad in enumerate(grads):
        if grad.numel():
            buckets.setdefault((grad.device, grad.dtype, grad.numel()), []).append(index)
    for (_, dtype, count), indices in buckets.items():
        block_size = 1024 if dtype == torch.float16 else 8192
        # Bound tuple specialization size and compilation cost for large models.
        for start in range(0, len(indices), 4):
            chunk = indices[start:start + 4]
            padded = chunk + [chunk[0]] * (4 - len(chunk))
            kernel[(((count + block_size - 1) // block_size) * len(chunk),)](
                *(grads[i] for i in padded), *(bufs[i] for i in padded),
                *(outputs[i] for i in padded), count, momentum1, momentum2, nesterov,
                BLOCK=block_size, enable_fp_fusion=False,
            )
    return outputs


def _apply_update_triton(params, updates, decay, alpha):
    """Apply updates to contiguous FP32 master parameters on the current stream."""
    if len(params) != len(updates):
        raise ValueError("Muon parameter and update counts must match")
    _validate_tensors([*params, *updates])
    for param, update in zip(params, updates):
        if param.dtype != torch.float32:
            raise ValueError("Muon triton parameter updates require FP32 master parameters")
        if param.shape != update.shape or param.device != update.device:
            raise ValueError("Muon parameters and updates must have matching shape and device")
    kernel = _npu_kernels()._apply_kernel  # pylint: disable=protected-access
    buckets = {}
    for index, (param, update) in enumerate(zip(params, updates)):
        if param.numel():
            buckets.setdefault((param.device, update.dtype, param.numel()), []).append(index)
    for (_, _, count), indices in buckets.items():
        for start in range(0, len(indices), 4):
            chunk = indices[start:start + 4]
            padded = chunk + [chunk[0]] * (4 - len(chunk))
            kernel[(((count + 8191) // 8192) * len(chunk),)](
                *(params[i] for i in padded), *(updates[i] for i in padded),
                count, decay, alpha, BLOCK=8192, enable_fp_fusion=False,
            )

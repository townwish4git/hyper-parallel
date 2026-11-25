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
"""Submit both Triton symmetric products of a Muon iteration through one torch-npu queue task."""

import logging
import os
import subprocess
from functools import lru_cache
from threading import Lock

import torch


_LOGGER = logging.getLogger(__name__)
_PLAN_LOCK = Lock()
# Plans retain compiled code and modules, never tensors or streams. The queue can still be using a plan after return.
_PLANS = {}


@lru_cache(maxsize=1)
def _runtime():
    if os.environ.get("TASK_QUEUE_ENABLE") == "0" or os.environ.get("TRITON_ENABLE_TASKQUEUE") == "0":
        return None
    # CPU/CUDA Muon and eager Triton on other backend versions do not require the optional native extension.
    try:
        from . import _muon_launcher as launcher
        from . import symmetric_matmul as backend
    except ImportError:
        return None
    try:
        return backend, launcher, launcher.load_host_extension()
    except (launcher.UnsupportedLauncher, ImportError, OSError, subprocess.CalledProcessError) as error:
        _LOGGER.warning("Using individual Muon Triton launches: %s", error)
        return None


def _strides(tensor):
    return (tensor.stride(0) if tensor.ndim == 3 else 0, tensor.stride(-2), tensor.stride(-1))


def _plan(runtime, lhs, rhs, out):
    key = (lhs.device, lhs.shape, lhs.stride(), rhs.stride())
    if key not in _PLANS:
        with _PLAN_LOCK:
            if key not in _PLANS:
                backend, launcher, extension = runtime
                n, k = lhs.shape[-2:]
                batch = lhs.size(0) if lhs.ndim == 3 else 1
                blocks = backend._block_shape(n, k, batch)
                counts = backend._core_counts(lhs.device)
                backend._check_indexing_range((n, k, batch), blocks, counts)
                cores = counts[0]
                with torch.npu.device(lhs.device):
                    compiled = backend._cv_matmul_kernel.warmup(lhs, rhs, out, n, k, batch, *_strides(lhs),
                                                                *_strides(rhs), *blocks, cores,
                                                                grid=(cores,), num_stages=2)
                    try:
                        _PLANS[key] = launcher.bind_product(extension, compiled, cores, lhs, rhs, out)
                    except launcher.UnsupportedLauncher as error:
                        _LOGGER.warning("Using individual Muon Triton launches for %s: %s", lhs.shape, error)
                        _PLANS[key] = (None,)
    return _PLANS[key][0]


def try_paired_iteration(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                         coefficients: tuple) -> bool:
    """Run all NS iterations on fresh optimizer buffers, or leave them untouched for the eager path."""
    if (x.device.type != "npu" or x.dtype != torch.bfloat16 or torch.is_grad_enabled() or not coefficients
            or x.ndim not in (2, 3) or x.numel() == 0 or not (x.is_contiguous() or x.mT.is_contiguous())):
        return False
    runtime = _runtime()
    if runtime is None:
        return False
    gram = _plan(runtime, x, x.mT, a)
    square = _plan(runtime, a, a, b)
    if gram is None or square is None:
        return False
    runtime[2].iterate(gram, square, x, a, b, c, coefficients)
    return True

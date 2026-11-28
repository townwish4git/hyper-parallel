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
"""Experimental Muon callback for the fused Ascend C Newton–Schulz step."""

from collections.abc import Callable, Sequence
from functools import lru_cache

import torch

from hyper_parallel.components.functional._ascendc.symmetric_matmul import ascendc_fused_ns_step_out
from hyper_parallel.core.optimizer.muon import zeropower_via_newtonschulz5


@lru_cache(maxsize=None)
def _is_fused_ns_device(device: torch.device) -> bool:
    """Identify devices supported by the mixed Cube/Vector kernel."""
    return torch.npu.get_device_name(device).lower().startswith(("ascend910b", "ascend910_93", "ascend950"))


def _supports_fused_ns(inputs: torch.Tensor) -> bool:
    if inputs.device.type != "npu" or inputs.dtype != torch.bfloat16 or inputs.ndim not in (2, 3):
        return False
    rows, cols = inputs.shape[-2:]
    # The device copies 64-element rows using a 16-bit stride in 32-byte blocks.
    return (
        0 < inputs.numel() <= 2**31 - 1
        and max(rows, cols) <= (2**16 - 1) * 16 + 64
        and rows % 64 == 0
        and cols % 64 == 0
        and _is_fused_ns_device(inputs.device)
    )


def make_ascendc_newton_schulz(
        coefficients: Sequence[tuple[float, float, float]], *, epsilon: float,
) -> Callable[..., torch.Tensor]:
    """Bind Muon's coefficients and normalization to the fused Ascend C kernel.

    Args:
        coefficients: One coefficient triple per Newton–Schulz step.
        epsilon: Value added to the Frobenius norm during input normalization.

    Returns:
        Muon zeropower callback using fused steps for supported BF16 NPU matrices.
    """
    schedule = tuple(coefficients)

    @torch.no_grad()
    def zeropower(inputs: torch.Tensor, steps: int) -> torch.Tensor:
        if not _supports_fused_ns(inputs):
            return zeropower_via_newtonschulz5(inputs, steps, "custom", epsilon, schedule)

        step_out = ascendc_fused_ns_step_out()
        transposed = inputs.size(-2) > inputs.size(-1)
        x = inputs.mT if transposed else inputs
        x = (x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)).contiguous()
        gram = torch.empty((*x.shape[:-1], x.size(-2)), dtype=x.dtype, device=x.device)
        poly = torch.empty_like(gram)
        out = torch.empty_like(x)
        for a, b, c in schedule[:steps]:
            step_out(x, a, b, c, gram=gram, poly=poly, out=out)
            x, out = out, x
        return x.mT if transposed else x

    return zeropower

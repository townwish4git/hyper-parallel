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
"""HyperParallel symmetric GEMM and fused Newton-Schulz operators for Ascend NPUs."""

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch


@lru_cache(maxsize=1)
def _load_ascendc_ops() -> None:
    """Load the shared device and Torch dispatcher libraries."""
    path = Path(__file__).with_name("libhp_symm_torch.so")
    if not path.is_file():
        raise RuntimeError("Build the Ascend C symmetric GEMM with _ascendc/symmetric_matmul/build.py")
    torch.ops.load_library(str(path.with_name("libhp_pipeline_kernel.so")))
    torch.ops.load_library(str(path))


def ascendc_symmetric_gemm_out() -> Callable[..., torch.Tensor]:
    """Load the queued FP16/BF16 Ascend C out operator for symmetric products."""
    _load_ascendc_ops()
    return torch.ops.hyper_parallel.symmetric_gemm_mirror.out


def ascendc_fused_ns_step_out() -> Callable[..., torch.Tensor]:
    """Load the fused BF16 Newton–Schulz step for Ascend A2, A3, and A5."""
    _load_ascendc_ops()
    operator = getattr(torch.ops.hyper_parallel, "fused_ns_step", None)
    if operator is None:
        raise RuntimeError("Rebuild HyperParallel for Ascend A2, A3, or A5 to register the fused Newton-Schulz step.")
    return operator.out

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
"""Queued triangular GEMM with symmetric writeback on Ascend NPUs."""

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch


@lru_cache(maxsize=1)
def ascendc_symmetric_gemm_out() -> Callable[..., torch.Tensor]:
    """Load the queued FP16/BF16 Ascend C out operator for symmetric products."""
    path = Path(__file__).with_name("libhp_symm_torch.so")
    if not path.is_file():
        raise RuntimeError("Build the Ascend C symmetric GEMM with _ascendc/symmetric_matmul/build.py")
    torch.ops.load_library(str(path.with_name("libhp_pipeline_kernel.so")))
    torch.ops.load_library(str(path))
    return torch.ops.hyper_parallel.symmetric_gemm_mirror.out

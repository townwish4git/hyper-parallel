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
"""Single-device launcher for the optional queued Ascend C operator."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_LIBRARY = (Path(__file__).resolve().parents[3] / "hyper_parallel/components/functional/_ascendc"
            / "symmetric_matmul/libhp_symm_torch.so")


@pytest.mark.skipif(not _LIBRARY.is_file(), reason="requires the compiled Ascend C symmetric GEMM")
@arg_mark(plat_marks=["platform_ascend", "platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_ascendc_symmetric_matmul() -> None:
    """Feature: Ascend C symmetric GEMM.

    Description: Execute Gram and square on a non-default NPU stream.
    Expectation: Results match native matmul, off-diagonal mirrors agree bitwise and guards remain intact.
    """
    worker = str(Path(__file__).with_name("_test_ascendc_symmetric_matmul.py"))
    with patch.dict(os.environ, {"CLOSE_MATMUL_K_SHIFT": "1", "TASK_QUEUE_ENABLE": "1"}):
        torchrun_case(worker, "test_ascendc_symmetric_matmul", num_proc=1)


@pytest.mark.skipif(not _LIBRARY.is_file(), reason="requires the compiled Ascend C libraries")
@arg_mark(plat_marks=["platform_ascend", "platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_ascendc_fused_ns_step() -> None:
    """Exercise fused Newton–Schulz on Ascend A2, A3, and A5."""
    worker = str(Path(__file__).with_name("_test_ascendc_symmetric_matmul.py"))
    with patch.dict(os.environ, {"CLOSE_MATMUL_K_SHIFT": "1", "TASK_QUEUE_ENABLE": "1"}):
        torchrun_case(worker, "test_ascendc_fused_ns_step", num_proc=1)

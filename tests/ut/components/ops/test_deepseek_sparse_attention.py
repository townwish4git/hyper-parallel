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

"""Native deepseek_sparse_attention forward and gradient precision contracts."""

import unittest

import torch
from torch.nn import functional as F

from hyper_parallel.components.ops import deepseek_sparse_attention_absorbed
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestDeepseekSparseAttentionPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("deepseek_sparse_attention", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_padded_selection_matches_gathered_selection(self) -> None:
        """Check packed suffixes with repeated IDs, masked slots and an empty row."""
        generator = torch.Generator().manual_seed(304)
        inputs = tuple(torch.randn(*shape, generator=generator, requires_grad=True) for shape in (
            (1, 5, 4, 8), (1, 9, 2, 8), (1, 5, 4, 4), (1, 9, 2, 4),
        ))
        query_ends = torch.tensor([2, 5], dtype=torch.int32)
        key_ends = torch.tensor([4, 9], dtype=torch.int32)
        indices = torch.tensor([[0, 0, 2], [-1, -1, -1], [0, 2, 3], [1, 3, 11], [0, 4, -1]], dtype=torch.int32)
        indices = indices.unsqueeze(1)
        expected = deepseek_sparse_attention_absorbed(*inputs, indices, 0.25, query_ends, key_ends)
        actual = deepseek_sparse_attention_absorbed(*inputs, F.pad(indices, (0, 5), value=-1),
                                                   0.25, query_ends, key_ends)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0][:, 1], torch.zeros_like(actual[0][:, 1]), rtol=0, atol=0)

        upstream = torch.randn(actual[0].shape, generator=generator)
        torch.testing.assert_close(torch.autograd.grad(actual[0], inputs, upstream),
                                   torch.autograd.grad(expected[0], inputs, upstream))

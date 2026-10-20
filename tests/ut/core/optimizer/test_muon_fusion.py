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

"""Numerical and storage contracts for optional Muon optimizations."""
import unittest

import torch

from hyper_parallel.core.optimizer.muon import Muon, zeropower_via_newtonschulz5
from tests.common.mark_utils import arg_mark


class TestMuonFusion(unittest.TestCase):
    """Use float64 to separate algebraic parity from low-precision rounding."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fused_ns_matches_reference(self):
        """Feature: Fused NS.

        Description: Compare all coefficient schedules and matrix batch shapes.
        Expectation: Float64 parity, distinct output storage and unchanged input.
        """
        torch.manual_seed(18)
        for shape in ((9, 17), (17, 9), (3, 9, 17), (2, 3, 17, 9)):
            for variant in ("legacy", "asym5", "custom"):
                with self.subTest(shape=shape, variant=variant):
                    value = torch.randn(shape, dtype=torch.float64)
                    original = value.clone()
                    coefficients = [(3.0, -4.0, 2.0)] * 5 if variant == "custom" else None
                    options = {"steps": 5, "ns_variant": variant, "ns_coefficients": coefficients}
                    expected = zeropower_via_newtonschulz5(value, **options)
                    actual = zeropower_via_newtonschulz5(value, implementation="standard_fused", **options)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(value, original, rtol=0, atol=0)
                    self.assertNotEqual(actual.data_ptr(), value.data_ptr())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fused_ns_zero_and_noncontiguous(self):
        """Feature: Fused NS input layouts.

        Description: Exercise zeros and transposed views.
        Expectation: Match the unfused polynomial.
        """
        for value in (torch.zeros(5, 8), torch.randn(5, 8).t()):
            expected = zeropower_via_newtonschulz5(value, 5)
            actual = zeropower_via_newtonschulz5(value, 5, implementation="standard_fused")
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_implementation(self):
        """Feature: NS configuration validation.

        Description: Select an unknown implementation.
        Expectation: Raise ValueError before optimizer initialization.
        """
        with self.assertRaisesRegex(ValueError, "Unknown NS"):
            Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_implementation="unknown")

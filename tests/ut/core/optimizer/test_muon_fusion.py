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
from hyper_parallel.core.optimizer.muon_pointwise import _momentum_update_triton
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


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_momentum_backend_validation(self):
        """Feature: Momentum backend configuration.

        Description: Reject unknown backends and conflicting callbacks.
        Expectation: Fail before modifying any optimizer state.
        """
        param = torch.nn.Parameter(torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, "Unknown momentum"):
            Muon([param], momentum_implementation="unknown")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            Muon([param], momentum_implementation="triton", momentum_update_fn=lambda *args: [])


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_momentum_rejects_cpu_without_mutation(self):
        """Feature: Explicit NPU kernel requirements.

        Description: Try the optional backend on CPU tensors.
        Expectation: Reject before mutating persistent state or importing Triton.
        """
        grad = torch.ones(2, 2)
        buf = torch.zeros_like(grad)
        with self.assertRaisesRegex(ValueError, "contiguous local NPU"):
            _momentum_update_triton([grad], [buf], .95, .85, True)
        torch.testing.assert_close(buf, torch.zeros_like(buf), atol=0, rtol=0)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scale_writeback_matches_native_path(self):
        """Feature: Fused scaling into NS views.

        Description: Compare ordinary, batched and transposed logical matrices.
        Expectation: Preserve the source API and match the unfused result.
        """
        for shape in ((8, 16), (16, 8), (3, 8, 16)):
            optimizer = object.__new__(Muon)
            optimizer.reshape_fn = None
            optimizer.ns_transform_fn = None
            optimizer.zeropower_fn = None
            param = torch.nn.Parameter(torch.ones(shape))
            original = torch.randn(shape)
            group = {"ns_steps": 5, "ns_variant": "asym5", "ns_coefficients": None,
                     "ns_epsilon": 1e-10, "matched_adamw_rms": .2, "zero_rms_scale_mode": "zero",
                     "apply_lr_in_update": False, "lr": .01}
            expected = optimizer._compute_batched_ns_updates_fast([param], {param: original.clone()}, group, True)
            group["fused_scale_writeback"] = True
            actual = optimizer._compute_batched_ns_updates_fast([param], {param: original.clone()}, group, True)
            torch.testing.assert_close(actual[param], expected[param], atol=0, rtol=0)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_gram_algebra_and_restarts(self):
        """Feature: Gram-space NS recurrence.

        Description: Compare both orientations, batches, coefficients and restarts in float64.
        Expectation: Match standard NS without modifying the supplied input.
        """
        torch.manual_seed(19)
        for shape in ((8, 32), (32, 8), (2, 8, 32), (8, 8)):
            for resets in ((), (2,)):
                for variant in ("legacy", "asym5"):
                    with self.subTest(shape=shape, resets=resets, variant=variant):
                        value = torch.randn(shape, dtype=torch.float64)
                        original = value.clone()
                        expected = zeropower_via_newtonschulz5(value, 5, ns_variant=variant)
                        actual = zeropower_via_newtonschulz5(
                            value, 5, ns_variant=variant, implementation="gram", gram_reset_iterations=resets)
                        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
                        torch.testing.assert_close(value, original, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_gram_zeros_and_configuration(self):
        """Feature: Gram stability and configuration.

        Description: Exercise a zero matrix and invalid precision or restart settings.
        Expectation: Return finite zeros and reject unsupported settings.
        """
        value = torch.zeros(8, 32)
        actual = zeropower_via_newtonschulz5(value, 5, implementation="gram")
        torch.testing.assert_close(actual, value, atol=0, rtol=0)
        for options in ({"gram_compute_dtype": "int8"}, {"gram_reset_iterations": (0,)}):
            with self.assertRaises(ValueError):
                Muon([torch.nn.Parameter(value)], ns_implementation="gram", **options)

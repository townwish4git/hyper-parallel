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
"""Independent symmetric GEMM opt-in, standard NS rounding and callback precedence."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer import _filter_optimizer_config
from hyper_parallel.core.optimizer.muon import Muon, zeropower_via_newtonschulz5

from tests.common.mark_utils import arg_mark


class TestMuonSymmetricGemm(unittest.TestCase):
    """Verify the public switch without distributed initialization or NPU hardware."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_yaml_builder_preserves_switch(self):
        """Feature: YAML switch forwarding.

        Description: Normalize prefixed and unprefixed configuration keys.
        Expectation: The explicit switch reaches the Muon constructor.
        """
        for key in ("use_symm_gemm", "muon_use_symm_gemm"):
            actual = _filter_optimizer_config("muon", Muon, {key: True})
            self.assertEqual(actual, {"use_symm_gemm": True})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_step_preparation_keeps_empty_and_owned_groups(self):
        """Feature: Optimizer step preparation.

        Description: Cover unsharded, empty and explicitly owned momentum inputs.
        Expectation: Counters advance once and empty owned groups skip momentum work.
        """
        optimizer = object.__new__(Muon)
        optimizer.param_groups = [{"step": None}, {"step": 3}, {}]
        optimizer._hsdp_assignment_batches = {1: {"no_comm": []}, 2: {"no_comm": ["owned"]}}
        optimizer.unshard_params_by_group = {0: ["plain"]}
        with patch.object(optimizer, "_update_muon_momentum", side_effect=lambda _, params: {p: p for p in params}):
            actual = optimizer._prepare_step_ns_inputs()
        self.assertEqual(actual, {0: {"plain": "plain"}, 1: {}, 2: {"owned": "owned"}})
        self.assertEqual([group["step"] for group in optimizer.param_groups], [1, 4, 1])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_cpu_fallback_preserves_standard_rounding_and_layout(self):
        """Feature: cpu fallback preserves standard rounding and layout.

        Description: Wide, tall and batched inputs keep the original coefficient roundings.
        Expectation: All stated numerical, routing or validation assertions hold.
        """
        torch.manual_seed(61)
        for shape in ((7, 19), (19, 7), (2, 7, 19), (2, 19, 7)):
            value = torch.randn(shape, dtype=torch.bfloat16)
            for variant, steps in (("asym5", 5), ("legacy", 3)):
                with self.subTest(shape=shape, variant=variant):
                    reference = zeropower_via_newtonschulz5(value, steps, ns_variant=variant)
                    actual = zeropower_via_newtonschulz5(value, steps, ns_variant=variant, use_symm_gemm=True)
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0, check_stride=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_disabled_switch_does_not_dispatch_symmetric_products(self):
        """Feature: disabled switch does not dispatch symmetric products.

        Description: Default execution keeps the original native matmul path.
        Expectation: All stated numerical, routing or validation assertions hold.
        """
        with patch("hyper_parallel.core.optimizer.muon.symmetric_matmul") as product:
            zeropower_via_newtonschulz5(torch.randn(7, 19), 5)
            product.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_switch_reaches_batched_ns_and_callback_takes_precedence(self):
        """Feature: switch reaches batched ns and callback takes precedence.

        Description: Forward the flag after batching; keep explicit custom NS callbacks authoritative.
        Expectation: All stated numerical, routing or validation assertions hold.
        """
        optimizer = object.__new__(Muon)
        optimizer.zeropower_fn = None
        inputs = [torch.randn(8, 16, dtype=torch.bfloat16) for _ in range(2)]
        with patch("hyper_parallel.core.optimizer.muon.zeropower_via_newtonschulz5",
                   side_effect=lambda value, **_: value.clone()) as ns:
            optimizer._compute_batched_ns_outputs_for_tensors(inputs, 5, use_symm_gemm=True)
            self.assertEqual(ns.call_args.args[0].shape, (2, 8, 16))
            self.assertTrue(ns.call_args.kwargs["use_symm_gemm"])
        optimizer.zeropower_fn = lambda value, steps: value.clone()
        with patch("hyper_parallel.core.optimizer.muon.zeropower_via_newtonschulz5") as ns:
            optimizer._compute_batched_ns_outputs_for_tensors(inputs, 5, use_symm_gemm=True)
            ns.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_flag_is_stored_in_parameter_groups_and_checkpoints(self):
        """Feature: flag is stored in parameter groups and checkpoints.

        Description: A saved opt-in survives optimizer restoration independently for each group.
        Expectation: All stated numerical, routing or validation assertions hold.
        """
        parameters = [torch.nn.Parameter(torch.ones(2, 4)) for _ in range(2)]
        optimizer = Muon([{"params": [parameters[0]], "use_symm_gemm": True},
                          {"params": [parameters[1]], "use_symm_gemm": False}])
        restored = Muon([{"params": [parameters[0]]}, {"params": [parameters[1]]}])
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual([group["use_symm_gemm"] for group in restored.param_groups], [True, False])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_invalid_flag_is_rejected(self):
        """Feature: invalid flag is rejected.

        Description: Reject truthy non-bool flags at both public boundaries.
        Expectation: All stated numerical, routing or validation assertions hold.
        """
        value = torch.zeros(7, 19, dtype=torch.bfloat16)
        for flag in ("true", 1, None):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "use_symm_gemm"):
                    zeropower_via_newtonschulz5(value, 5, use_symm_gemm=flag)
                with self.assertRaisesRegex(ValueError, "use_symm_gemm"):
                    Muon([torch.nn.Parameter(value)], use_symm_gemm=flag)

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
"""Regression tests for cropped Qwen3-MoE Trainer build options."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import yaml

from examples.training_demo import cropped_qwen3_moe
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark


class TestCroppedQwen3Moe(unittest.TestCase):
    """Check the demo accepts and forwards Trainer-owned runtime options."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_yaml_target_forwards_trainer_options(self) -> None:
        """Build both YAML model targets with default and enabled input swapping.

        Feature: Cropped Qwen3-MoE Trainer model construction.
        Description: Build each YAML model target with Trainer runtime options.
        Expectation: Options reach from_config and the model retains four layers.
        """
        demo_dir = Path(cropped_qwen3_moe.__file__).parent
        for filename in ("train_parallel_offline.yaml", "train_parallel_online.yaml"):
            with (demo_dir / filename).open(encoding="utf-8") as stream:
                model_kwargs = yaml.safe_load(stream)["model"]
            target_path = model_kwargs.pop("_target_")
            target = Target(
                cropped_qwen3_moe.build_cropped_qwen3_moe,
                target_path=target_path,
                **model_kwargs,
            )
            for swap_inputs, model_init_dtype in ((False, None), (True, "float32")):
                with self.subTest(filename=filename, swap_inputs=swap_inputs):
                    config = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=48, use_cache=True)
                    runtime_options = {
                        "distributed_setup": sentinel.distributed_setup,
                        "peft_config": None,
                        "activation_checkpoint": "full" if swap_inputs else "off",
                        "swap_inputs": swap_inputs,
                        "activation_swap": "none",
                        "compile_config": CompileConfig(),
                        "model_init_dtype": model_init_dtype,
                    }
                    with (
                        patch.object(cropped_qwen3_moe.AutoConfig, "from_pretrained", return_value=config) as load,
                        patch.object(cropped_qwen3_moe.HyperAutoModelForCausalLM, "from_config") as build,
                    ):
                        model = target.build(**runtime_options)

                    self.assertIs(model, build.return_value)
                    self.assertEqual(config.num_hidden_layers, 4)
                    self.assertFalse(config.use_cache)
                    load.assert_called_once_with(
                        model_kwargs["config_path"], local_files_only=True, trust_remote_code=False,
                    )
                    build.assert_called_once_with(
                        config,
                        torch_dtype="bfloat16",
                        attn_implementation="sdpa",
                        validate_placement=False,
                        **runtime_options,
                    )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_grouped_options_forward_runtime_overrides(self) -> None:
        """Allow explicit keyword values to override grouped swap and dtype options.

        Feature: Grouped cropped Qwen3-MoE build options.
        Description: Override grouped input swapping and model dtype via keywords.
        Expectation: from_config receives overrides without mutating grouped options.
        """
        config = SimpleNamespace(model_type="qwen3_moe")
        options = cropped_qwen3_moe.CroppedQwen3MoeOptions(swap_inputs=False, model_init_dtype="bfloat16")
        with (
            patch.object(cropped_qwen3_moe.AutoConfig, "from_pretrained", return_value=config),
            patch.object(cropped_qwen3_moe.HyperAutoModelForCausalLM, "from_config") as build,
        ):
            cropped_qwen3_moe.build_cropped_qwen3_moe(
                "/local/model", options=options, swap_inputs=True, model_init_dtype="float32",
            )

        self.assertTrue(build.call_args.kwargs["swap_inputs"])
        self.assertEqual(build.call_args.kwargs["model_init_dtype"], "float32")
        self.assertFalse(options.swap_inputs)
        self.assertEqual(options.model_init_dtype, "bfloat16")

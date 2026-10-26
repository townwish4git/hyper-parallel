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
"""Native Qwen RoPE recovery through the adapter and pretrained lifecycle."""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest
import torch
from torch import nn

from hyper_parallel import init_empty_weights
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models._transformers.model_builder import _apply_materialization_adapter
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state

pytest.importorskip("transformers", minversion="5.5.4")

# Native HF accelerator discovery must not initialize hardware in CPU contracts.
with patch("transformers.utils.is_torch_npu_available", return_value=False), patch(
    "transformers.utils.import_utils.is_torch_npu_available", return_value=False
):
    from transformers import (
        AutoModelForCausalLM,
        Qwen3Config,
        Qwen3MoeConfig,
        Qwen3_5TextConfig,
        Qwen3_5MoeTextConfig,
    )


def _configs() -> tuple:
    """Return small native configurations covering dense and MoE text RoPE."""
    common = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "use_cache": False}
    rotary = {"rope_type": "default", "rope_theta": 1000000.0}
    moe = {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32}
    hybrid = {"linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_num_key_heads": 2,
              "linear_num_value_heads": 2, "layer_types": ["linear_attention", "full_attention"]}
    return (
        Qwen3Config.from_dict({**common, "rope_parameters": dict(rotary)}),
        Qwen3MoeConfig.from_dict({**common, **moe, "rope_parameters": dict(rotary)}),
        Qwen3_5TextConfig.from_dict({**common, **hybrid,
                                    "rope_parameters": {**rotary, "partial_rotary_factor": 0.5}}),
        Qwen3_5MoeTextConfig.from_dict({**common, **hybrid, **moe,
                                       "rope_parameters": {**rotary, "partial_rotary_factor": 0.5}}),
    )


class TestQwenRopeMaterialization(unittest.TestCase):
    """Exercise real native modules rather than faking buffer initialization."""

    def setUp(self) -> None:
        """Keep CPU workloads small without leaking thread or RNG state."""
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(42)

    def test_pretrained_restores_rope_without_changing_loaded_parameters(self) -> None:
        """Native checkpoints must restore derived values as well as every saved parameter."""
        for config in _configs():
            with self.subTest(model_type=config.model_type), tempfile.TemporaryDirectory() as directory:
                native = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
                native.save_pretrained(directory)
                with patch("hyper_parallel.models._transformers.auto_model._current_device",
                           return_value=torch.device("cpu")):
                    candidate = HyperAutoModelForCausalLM.from_pretrained(
                        directory, torch_dtype=torch.float32, force_hf=True,
                        local_files_only=True, attn_implementation="sdpa")
                report = candidate._hp_checkpoint_load_report  # pylint: disable=protected-access
                self.assertEqual(report.missing_keys, ())
                self.assertEqual(report.unexpected_keys, ())
                reference_parameters = dict(native.named_parameters())
                for name, parameter in candidate.named_parameters():
                    torch.testing.assert_close(parameter, reference_parameters[name], rtol=0, atol=0)
                expected_rope = native.model.rotary_emb
                actual_rope = candidate.model.rotary_emb
                for name in ("inv_freq", "original_inv_freq"):
                    torch.testing.assert_close(getattr(actual_rope, name), getattr(expected_rope, name), rtol=0, atol=0)
                    self.assertNotIn(f"model.rotary_emb.{name}", candidate.state_dict())
                positions = torch.arange(4).unsqueeze(0)
                x = torch.zeros(1, 4, config.hidden_size)
                expected_cos, expected_sin = expected_rope(x, positions)
                actual_cos, actual_sin = actual_rope(x, positions)
                torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
                torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)

    def test_text_and_vision_rebuild_preserves_checkpoint_state_and_buffer_identity(self) -> None:
        """Declared buffers recover in place on both init paths, including native vision RoPE."""
        for config in _configs():
            with self.subTest(model_type=config.model_type):
                with init_empty_weights():
                    native = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
                model = nn.Module()
                model.config = config
                model.text_rope = native.model.rotary_emb
                model.weight = nn.Parameter(torch.empty(2, device="meta"))
                model.register_buffer("persistent", torch.empty(2))
                if config.model_type.startswith("qwen3_5"):
                    family = "qwen3_5_moe" if "moe" in config.model_type else "qwen3_5"
                    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
                    class_name = (
                        "Qwen3_5MoeVisionRotaryEmbedding" if "moe" in family else "Qwen3_5VisionRotaryEmbedding"
                    )
                    model.vision_rope = getattr(module, class_name)(dim=16)
                expected = {name: value.clone() for name, value in model.named_buffers() if name != "persistent"}
                _apply_materialization_adapter(model)
                model.to_empty(device="cpu")
                with torch.no_grad():
                    model.weight.fill_(7)
                    model.persistent.fill_(9)
                    for name in expected:
                        model.get_buffer(name).fill_(-99)
                identities = {name: id(value) for name, value in model.named_buffers()}
                parameter = model.weight
                for reason in ("random_init", "checkpoint_load", "checkpoint_load"):
                    rebuild_materialized_state(model, MaterializationContext(reason=reason, device=torch.device("cpu")))
                    for name, value in expected.items():
                        torch.testing.assert_close(model.get_buffer(name), value, rtol=0, atol=0)
                        self.assertEqual(id(model.get_buffer(name)), identities[name])
                        self.assertNotIn(name, model.state_dict())
                    self.assertIs(model.weight, parameter)
                    torch.testing.assert_close(model.weight, torch.full((2,), 7.0), rtol=0, atol=0)
                    torch.testing.assert_close(model.persistent, torch.full((2,), 9.0), rtol=0, atol=0)

    def test_registry_discovery_is_lazy_and_finds_text_aliases_first(self) -> None:
        """Fresh discovery must find text-only families without importing native implementations."""
        script = """
import sys
from hyper_parallel.models.registry import get_model_adapter
for identity in ('qwen3_5_moe_text', 'qwen3_5_moe', 'qwen3_5_text', 'qwen3_5', 'qwen3_moe', 'qwen3'):
    spec = get_model_adapter(identity)
    assert spec is not None and callable(spec.materialization), identity
for family in ('qwen3', 'qwen3_moe', 'qwen3_5', 'qwen3_5_moe'):
    assert f'transformers.models.{family}.modeling_{family}' not in sys.modules, family
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[4],
                                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

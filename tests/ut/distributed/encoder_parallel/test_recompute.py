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
"""Independent vision checkpoint selection and per-microbatch CP replay."""

from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed.activation_checkpoint import apply_vision_activation_checkpointing
from hyper_parallel.distributed.encoder_parallel.attention import (
    EncoderAttentionContext,
    activate_encoder_attention,
    current_encoder_attention_context,
)
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderLocalInputs, EncoderWorkItem
from hyper_parallel.distributed.encoder_parallel.planning import build_segments
from hyper_parallel.models.build_options import ModelBuildOptions, VisionActivationCheckpointConfig
from hyper_parallel.models._transformers.model_builder import _apply_activation_features
from hyper_parallel.models.deepseek_v41.adapter import registration  # pylint: disable=unused-import
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import (
    DeepseekV41EncoderParallelAttention,
    DeepseekV41VisionBridge,
)
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionAligner, DeepseekV41VisionTower
from hyper_parallel.trainer.config.parallelism import ActivationCheckpointConfig
from hyper_parallel.trainer.config.parser import load_training_config
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.common.mark_utils import arg_mark


def _make_model() -> nn.Module:
    """Construct a small native facade with the real visual attention replacement."""
    config = SimpleNamespace(
        v41_vision_hidden_size=8, v41_vision_patch_size=2, v41_vision_num_attention_heads=2,
        v41_vision_rope_theta=10000.0, v41_vision_intermediate_size=16,
        v41_vision_num_hidden_layers=2, v41_vision_downsample_ratio=1, hidden_size=8,
        model_type="deepseek_v41",
    )
    model = nn.Module()
    model.config = config
    model.model = nn.Module()
    model.model.vision = DeepseekV41VisionTower(config)
    model.model.aligner = DeepseekV41VisionAligner(config)
    for index, block in enumerate(model.model.vision.blocks):
        block.attn = DeepseekV41EncoderParallelAttention.from_source(
            block.attn, module_fqn=f"model.vision.blocks.{index}.attn",
        )
    return model


class TestVisionRecompute(unittest.TestCase):
    """Exercise real non-reentrant backward without distributed initialization."""

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_vision_config_is_independent_and_normalizes_programmatic_options(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: Decoder off and vision full survive the typed YAML and programmatic boundaries.
        Expectation: Vision configuration normalizes without enabling decoder recompute.
        """
        config = resolve_component(
            {"mode": "off", "vision": {"mode": "full", "layer_indices": [1]}},
            annotation=ActivationCheckpointConfig, path="activation_checkpoint",
        )
        self.assertEqual(config.mode, "off")
        self.assertEqual(config.vision, VisionActivationCheckpointConfig(mode="full", layer_indices=[1]))
        options = ModelBuildOptions(vision_activation_checkpoint={"mode": "full"})
        self.assertEqual(options.vision_activation_checkpoint.mode, "full")
        self.assertIsNone(options.activation_checkpoint)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_native_tower_forward_backward_matches_without_encoder_parallel(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: Vision checkpointing also works when no Encoder-CP execution is active.
        Expectation: Outputs and every used parameter gradient match the native tower.
        """
        reference = _make_model()
        candidate = deepcopy(reference)
        original_keys = set(candidate.state_dict())
        apply_vision_activation_checkpointing(candidate, {"mode": "full"})
        self.assertEqual(set(candidate.state_dict()), original_keys)
        pixels = torch.randn(5, 3, 2, 2)
        expected = reference.model.vision(pixels, 1, 5)
        actual = candidate.model.vision(pixels, 1, 5)
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        self._assert_gradients(candidate, reference)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_checkpoint_selection_keeps_original_parameter_paths(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: Select one visual block and preserve state/optimizer parameter names.
        Expectation: Only the selected block is wrapped and canonical names remain stable.
        """
        model = _make_model()
        parameter_names = tuple(name for name, _ in model.named_parameters())
        apply_vision_activation_checkpointing(model, {"mode": "full", "layer_indices": [1]})
        self.assertNotIsInstance(model.model.vision.blocks[0], CheckpointWrapper)
        self.assertIsInstance(model.model.vision.blocks[1], CheckpointWrapper)
        self.assertEqual(tuple(name for name, _ in model.named_parameters()), parameter_names)
        DeepseekV41VisionBridge(model).validate_training({"micro_batch_size": 1})

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_backward_restores_each_microbatch_geometry_and_separate_audit(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: Two live graphs replay their own segments and restore an enclosing context.
        Expectation: Each replay restores its own context and leaves original audits unchanged.
        """
        reference = _make_model()
        candidate = deepcopy(reference)
        apply_vision_activation_checkpointing(candidate, {"mode": "full"})
        bridge = DeepseekV41VisionBridge(candidate)
        expected_names = frozenset(f"model.vision.blocks.{index}.attn" for index in range(2))
        mesh = SimpleNamespace(size=lambda: 1)
        seen = []

        def _record(name: str, _module: nn.Module, _args: tuple[Any, ...]) -> None:
            """Record each block's original or replay geometry."""
            context = current_encoder_attention_context()
            seen.append((name, is_recomputing(), context.segments[0].item.sequence_length))

        for index, block in enumerate(candidate.model.vision.blocks):
            block.attn.register_forward_pre_hook(partial(_record, f"model.vision.blocks.{index}.attn"))
        candidate_losses = []
        reference_losses = []
        contexts = []
        module = "hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel"
        with patch(f"{module}.ulysses_seq_to_head", side_effect=lambda tensor, **_kwargs: tensor):
            with patch(f"{module}.ulysses_head_to_seq", side_effect=lambda tensor, **_kwargs: tensor):
                for count in (5, 3):
                    pixels = torch.randn(count, 3, 2, 2)
                    item = EncoderWorkItem(count, 0, count, count, 0, count + 3, 0, count,
                                           {"vit_hw": (1, count), "llm_hw": (1, count)})
                    segments = build_segments((item,), 0, 1)
                    context = EncoderAttentionContext(segments, mesh, bridge.head_counts, expected_names)
                    with activate_encoder_attention(context):
                        actual = bridge.encode_local(EncoderLocalInputs(pixels), segments)
                    context.assert_exactly_once()
                    contexts.append(context)
                    expected = reference.model.vision(pixels, 1, count)
                    torch.testing.assert_close(actual, expected)
                    candidate_losses.append(actual.square().sum())
                    reference_losses.append(expected.square().sum())
                enclosing = EncoderAttentionContext((), mesh, {}, frozenset())
                with activate_encoder_attention(enclosing):
                    sum(candidate_losses).backward()
                    self.assertIs(current_encoder_attention_context(), enclosing)
        sum(reference_losses).backward()
        self._assert_gradients(candidate, reference)
        self.assertIsNone(current_encoder_attention_context())
        for context in contexts:
            context.assert_exactly_once()
        for count in (5, 3):
            for name in expected_names:
                self.assertEqual(seen.count((name, False, count)), 1)
                self.assertEqual(seen.count((name, True, count)), 1)

    @staticmethod
    def _assert_gradients(candidate: nn.Module, reference: nn.Module) -> None:
        """Compare every trainable visual parameter including non-recomputed units."""
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            expected = expected_parameters[name].grad
            if expected is None:
                if parameter.grad is not None:
                    raise AssertionError(f"unused parameter gained a gradient: {name}")
            else:
                torch.testing.assert_close(parameter.grad, expected, msg=lambda message: f"{name}: {message}")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_unsupported_and_invalid_options_fail_before_execution(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: Reject invalid indices, unsupported modes, compile, and swap combinations.
        Expectation: Invalid modes, indices, and unsupported execution combinations raise.
        """
        for options in (
                {"mode": "selective"}, {"mode": "off", "layer_indices": [0]},
                {"mode": "full", "layer_indices": []}, {"mode": "full", "layer_indices": [True]},
                {"mode": "full", "layer_indices": [-1]}, {"mode": "full", "layer_indices": [0, 0]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VisionActivationCheckpointConfig(**options)
        model = _make_model()
        for options in ({"layer_indices": [2]},):
            with self.assertRaisesRegex(ValueError, "exceeds maximum"):
                apply_vision_activation_checkpointing(model, {"mode": "full", **options})
        for execution in ({"enable_compile": True}, {"activation_swap": "attention"}):
            with self.subTest(execution=execution), self.assertRaisesRegex(ValueError, "requires compile"):
                apply_vision_activation_checkpointing(model, {"mode": "full"}, **execution)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_unmanaged_vision_checkpoint_is_still_rejected(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: A manual wrapper without CP replay support cannot silently enter the runtime.
        Expectation: The bridge rejects a wrapper without encoder replay support.
        """
        model = _make_model()
        model.model.vision.blocks[0] = checkpoint_wrapper(model.model.vision.blocks[0])
        with self.assertRaisesRegex(RuntimeError, "vision activation checkpoint is unsupported"):
            DeepseekV41VisionBridge(model).validate_training({"micro_batch_size": 1})

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_model_builder_applies_vision_while_decoder_is_off(self) -> None:
        """Feature: independent visual activation checkpointing.

        Description: The model build stage honors vision full independently of decoder off.
        Expectation: Vision blocks are wrapped while decoder recomputation stays off.
        """
        model = _make_model()
        result = _apply_activation_features(
            model, "off", "none", False, None,
            vision_activation_checkpoint=VisionActivationCheckpointConfig(mode="full"),
        )
        self.assertIs(result, model)
        self.assertTrue(all(isinstance(block, CheckpointWrapper) for block in model.model.vision.blocks))

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "allcards", "essential")
    def test_example_config_resolves_independent_vision(self) -> None:
        """Feature: usable vision recompute training recipe.

        Description: Resolve the full Encoder Parallel example through typed configuration.
        Expectation: Decoder off and vision full survive recipe resolution and CLI overrides.
        """
        root = Path(__file__).resolve().parents[4]
        recipe = root / "examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_encoder_parallel_e0.yaml"
        config = load_training_config(recipe, ["--activation_checkpoint.vision.layer_indices=[0,1]"])
        self.assertEqual(config.activation_checkpoint.mode, "off")
        self.assertEqual(config.activation_checkpoint.vision.mode, "full")
        self.assertEqual(config.activation_checkpoint.vision.layer_indices, [0, 1])

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
"""GLM training semantics for packed KDA, KPool distillation and label alignment."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import pytest
import torch

hf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

# GLM is optional on releases of Transformers that precede the native model.
# pylint: disable=wrong-import-position
from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.components.ops import ops
from hyper_parallel.models.glm5_next.adapter.build import load_config
from hyper_parallel.models.glm5_next.adapter.data import Glm5NextBatch
from hyper_parallel.models.glm5_next.adapter.replacements import GlmMhc, GlmVisionAttention
from hyper_parallel.models.glm5_next.modeling_glm5_next import DecoderInput, Glm5NextPackedKDA, construct_model
from hyper_parallel.models.glm5_next.sparse_attention import DocumentFragment, Glm5NextSparseAttention


_CONFIG = Path(__file__).resolve().parents[5] / "examples/training_demo/glm5_next/config_a5.json"


class TestGlmTraining(unittest.TestCase):
    """Compare native component and boundary behavior without requiring an NPU."""

    def setUp(self) -> None:
        self.config = load_config(str(_CONFIG))
        self.old_backends = ops.get_backends()
        ops.set_backends({
            "attention": "native", "deepseek_sparse_attention": "native", "dsa_indexer": "native", "kda": "native",
        })
        self.addCleanup(ops.set_backends, self.old_backends)
        self.generator = torch.Generator().manual_seed(713)

    def test_crop_structure_and_checkpoint_names(self) -> None:
        """Check all structural roles and the independent published output projection."""
        with torch.device("meta"):
            model = construct_model(self.config)
        layers = model.model.language_model.layers
        self.assertEqual(len(layers), 46)
        self.assertEqual(len(model.model.visual.blocks), 24)
        self.assertEqual(sum(isinstance(layer.self_attn, Glm5NextPackedKDA) for layer in layers), 34)
        self.assertEqual(sum(isinstance(layer.self_attn, Glm5NextSparseAttention) for layer in layers), 12)
        self.assertEqual(sum(isinstance(layer.mlp, hf.Glm5NextTextMoE) for layer in layers), 43)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 371_822_658)
        self.assertIsNot(model.lm_head.weight, model.get_input_embeddings().weight)
        state = model.state_dict()
        self.assertIn("model.language_model.layers.45.shared_head.norm.weight", state)
        self.assertNotIn("model.visual.rotary_pos_emb.inv_freq", state)
        mapped_targets = {name for rule in get_model_conversion_mapping(model, add_legacy=False)
                          for name in rule.target_patterns}
        self.assertIn("self_attn.conv1d.weight", mapped_targets)
        self.assertIn("mlp.experts.gate_up_proj", mapped_targets)

    def test_align_two_targets_before_cp_slicing(self) -> None:
        """Left padding and the end of each document must mask future predictions."""
        labels = torch.tensor([[-100, 1, 2, 3, 4, 5, 6, 7]])
        valid = labels.ne(-100)
        batch = Glm5NextBatch._normalize_source_batch({  # pylint: disable=protected-access
            "input_ids": labels.clamp_min(0), "labels": labels, "attention_mask": valid,
            "cu_seq_lens": torch.tensor([0, 4, 8]),
        })
        torch.testing.assert_close(batch["shift_labels"], torch.tensor([[-100, 2, 3, -100, 5, 6, 7, -100]]))
        torch.testing.assert_close(batch["mtp_labels"], torch.tensor([[-100, 3, -100, -100, 6, 7, -100, -100]]))

    def test_sparse_matches_hf_and_distills_only_indexer(self) -> None:
        """HF parity for selection with competing pools and gradient isolation for each training mode."""
        config = self.config.text_config
        config.index_topk = 8
        reference = hf.Glm5NextTextAttention(config, 3)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=0.1, generator=self.generator)
        reference.next_skip_topk = True
        candidate = Glm5NextSparseAttention(config, 3)
        candidate.load_state_dict(reference.state_dict())
        x = torch.randn(1, 17, getattr(config, "hidden_size"), generator=self.generator, requires_grad=True)
        expected = reference(x, attention_mask=torch.ones_like(x[..., 0], dtype=torch.bool))[0]
        args = {"fragments": (DocumentFragment(0, 17, 0),), "local_start": 0, "cp": None}
        actual = candidate(x, indexer_mode="frozen", **args).hidden
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        dy = torch.randn(actual.shape, generator=self.generator)
        torch.testing.assert_close(torch.autograd.grad(actual, x, dy)[0], torch.autograd.grad(expected, x, dy)[0],
                                   rtol=2e-4, atol=2e-5)

        for mode in ("dense_warmup", "sparse"):
            with self.subTest(mode=mode):
                result = candidate(x, indexer_mode=mode, **args)
                self.assertEqual(result.indexer_count.item(), 10)
                self.assertGreater(result.indexer_loss.item(), 0)
                bank = candidate.indexer
                parameters = (bank.wq_b.weight, bank.wk.weight, bank.weights_proj.weight,
                              bank.index_kpool_compress_gate, bank.index_kpool_compress_ape, x,
                              candidate.kv_a_proj_with_mqa.weight)
                grads = torch.autograd.grad(result.indexer_loss, parameters, allow_unused=True)
                self.assertTrue(all(grad.isfinite().all() and grad.norm() > 0 for grad in grads[:5]))
                self.assertTrue(all(grad is None or torch.count_nonzero(grad) == 0 for grad in grads[-2:]))

    def test_packed_kda_resets_at_document_boundary(self) -> None:
        """Compare one packed HP KDA call with independent HF document forwards."""
        config = self.config.text_config
        candidate = Glm5NextPackedKDA(config, 0)
        with torch.no_grad():
            for parameter in candidate.parameters():
                parameter.normal_(std=0.05, generator=self.generator)
        reference = hf.Glm5NextTextLinearAttention(config, 0)
        reference.load_state_dict(candidate.state_dict())
        x = torch.randn(1, 17, getattr(config, "hidden_size"), generator=self.generator, requires_grad=True)
        geometry = DecoderInput((0, 9, 17), (), 0, 17)
        actual = candidate(x, geometry=geometry, cp=None)
        with patch.object(hf, "chunk_kimi_delta_attention", hf.recurrent_kimi_delta_attention):
            expected = torch.cat((reference(x[:, :9]), reference(x[:, 9:])), dim=1)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
        dy = torch.randn(actual.shape, generator=self.generator)
        torch.testing.assert_close(torch.autograd.grad(actual, x, dy)[0], torch.autograd.grad(expected, x, dy)[0],
                                   rtol=3e-4, atol=3e-5)

    def test_optimized_vision_and_mhc_keep_reference_math(self) -> None:
        """Use identical parameters for per-frame vision attention and four-stream mixing."""
        source = hf.Glm5NextVisionAttention(self.config.vision_config)
        vision = GlmVisionAttention(deepcopy(source))
        hidden = torch.randn(13, 128, generator=self.generator, requires_grad=True)
        angle = torch.randn(13, 32, generator=self.generator).repeat(1, 2)
        boundaries = torch.tensor([0, 5, 13], dtype=torch.int32)
        args = (hidden, boundaries, (angle.cos(), angle.sin()))
        torch.testing.assert_close(vision(*args), source(*args), rtol=2e-4, atol=2e-5)

        connection = hf.Glm5NextTextHyperConnection(self.config.text_config)
        with torch.no_grad():
            for parameter in connection.parameters():
                parameter.normal_(std=0.05, generator=self.generator)
        mhc = GlmMhc(module=deepcopy(connection))
        residual = torch.randn(1, 5, 4, 128, generator=self.generator)
        post, combine, collapse = connection(residual)
        fused_post, fused_combine, fused_collapse = mhc(residual)
        torch.testing.assert_close((fused_post, fused_combine, fused_collapse), (post, combine, collapse))
        expected = post.unsqueeze(-1) * collapse.unsqueeze(-2) + combine.transpose(-1, -2) @ residual
        torch.testing.assert_close(mhc.combine_residual(residual, fused_collapse, fused_post, fused_combine), expected)
        mhc.reset_parameters()
        torch.testing.assert_close(mhc.scale, torch.ones_like(mhc.scale))
        torch.testing.assert_close(mhc.base, torch.zeros_like(mhc.base))

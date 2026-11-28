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
"""Flash-Next module parity and distributed gradient probes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpModel,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRotaryEmbedding,
)

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.qwen4_exp.adapter.crop import build_crop, cropped_config
from hyper_parallel.models.validation_spec import (
    DataValidationSpec,
    ModelValidationSpec,
    ModuleParityCase,
    ParameterProbeSpec,
    StateInvariantSpec,
    TopologyConstraint,
)

from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from hyper_parallel.trainer.config.parser import load_training_config


class _ReferenceGdn(nn.Module):
    """Execute the native GDN on ungrouped views of the candidate-format parameters."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.native = Qwen4ExpTextGatedDeltaNet(config, 0)
        width = self.native.head_k_dim
        value_width = self.native.value_dim // self.native.num_k_heads
        self.row_groups = (width, width, value_width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore Q, K and V row groups before calling the original HF module."""
        native_weights = {}
        for name in ("in_proj_qkv.weight", "conv1d.weight"):
            weight = self.native.get_parameter(name)
            groups = weight.unflatten(0, (self.native.num_k_heads, -1))
            native_weights[name] = torch.cat([rows.flatten(0, 1) for rows in groups.split(self.row_groups, dim=1)])
        return functional_call(self.native, native_weights, (hidden_states,))


class _ReferenceBackbone(nn.Module):
    """Expose backbone activations independently of the training facade."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.native = Qwen4ExpModel(config)

    def forward(self, **kwargs: Any) -> Any:
        """Execute the HF vision and text model."""
        return self.native(**kwargs)


class _ReferenceDraft(nn.Module):
    """Compose the published MTP fusion equation with independent HF decoder layers."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(config.hc_count * width, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(width, eps=config.rms_norm_eps)
        self.fc_embedding = nn.Linear(width, width, bias=False)
        self.fc_hidden = nn.Linear(width, width, bias=False)
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(config, 0)])
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)

    def forward(self, hidden_states, next_embeddings, position_ids, attention_mask):
        """Fuse each residual stream separately, then apply the draft block and mixer."""
        streams = self.pre_fc_norm_hidden(hidden_states).chunk(self.config.hc_count, dim=-1)
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(next_embeddings))
        fused = torch.cat([F.linear(stream, self.fc_hidden.weight) + embedding for stream in streams], dim=-1)
        decoded = self.layers[0](fused, self.rotary_emb(fused, position_ids), attention_mask=attention_mask)
        return self.hyper_connection_mixer(decoded)


def _config(context: Any) -> Any:
    config = cropped_config(str(context.manifest.reference_path), sparse_budget=8)
    config._attn_implementation = "sdpa"  # pylint: disable=protected-access
    config._experts_implementation = "eager"  # pylint: disable=protected-access
    return config


def _candidate(context: Any) -> nn.Module:
    torch.manual_seed(42)
    model = build_crop(str(context.manifest.reference_path), sparse_budget=8, model_init_dtype="float32")
    return model.to(device=context.device, dtype=getattr(torch, context.dtype))


def _candidate_gdn(context: Any) -> nn.Module:
    torch.manual_seed(42)
    config = load_training_config(context.manifest.launcher_config_path)
    setup = DistributedSetup()
    normalize_distributed_setup_overrides(setup, config)
    model = build_crop(
        str(context.manifest.reference_path), sparse_budget=8,
        model_init_dtype="float32", distributed_setup=setup,
    )
    return model.to(device=context.device, dtype=getattr(torch, context.dtype))


def _gdn_reference(context: Any) -> nn.Module:
    return _ReferenceGdn(_config(context).text_config).to(device=context.device, dtype=getattr(torch, context.dtype))


def _gdn_inputs(context: Any) -> torch.Tensor:
    values = torch.linspace(-1, 1, 16 * 64, device=context.device, dtype=getattr(torch, context.dtype))
    return values.reshape(1, 16, 64).requires_grad_()


def _reference_backbone(context: Any) -> nn.Module:
    return _ReferenceBackbone(_config(context)).to(device=context.device, dtype=getattr(torch, context.dtype))


def _reference_draft(context: Any) -> nn.Module:
    config = deepcopy(_config(context).text_config)
    config.num_hidden_layers = 1
    config.layer_types = ["indexed_attention"]
    config.ple_layer_ids = []
    config.rope_parameters["rope_theta"] = config.mtp["rope_theta"]
    reference = _ReferenceDraft(config)
    return reference.to(device=context.device, dtype=getattr(torch, context.dtype))


def _copy_native(reference: nn.Module, candidate: nn.Module, _context: Any) -> dict[str, str]:
    reference.native.load_state_dict(candidate.state_dict(), strict=True)
    return {name: f"native.{name}" for name, _ in candidate.named_parameters()}


def _copy_draft(reference: nn.Module, candidate: nn.Module, _context: Any) -> None:
    reference.load_state_dict(candidate.state_dict(), strict=True)


def _backbone_inputs(context: Any) -> dict[str, Any]:
    ids = torch.tensor([[11, 502, 500, 503, 12, 13, 14, 23, 24, 25, 23, 24]], device=context.device)
    pixels = torch.linspace(0, 1, 6144, device=context.device, dtype=getattr(torch, context.dtype))
    return {
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "pixel_values": pixels.reshape(4, -1).requires_grad_(),
        "image_grid_thw": torch.tensor([[1, 2, 2]], device=context.device),
        "mm_token_type_ids": ids.eq(500).int(), "use_cache": False,
    }


def _draft_inputs(context: Any) -> tuple[Any, ...]:
    dtype = getattr(torch, context.dtype)
    hidden = torch.linspace(-1, 1, 12 * 256, device=context.device, dtype=dtype).reshape(1, 12, 256)
    embedding = torch.linspace(1, -1, 12 * 64, device=context.device, dtype=dtype).reshape(1, 12, 64)
    positions = torch.arange(12, device=context.device).view(1, 1, -1).expand(3, -1, -1)
    mask = torch.ones(1, 1, 12, 12, device=context.device, dtype=torch.bool).tril()
    return hidden.requires_grad_(), embedding.requires_grad_(), positions, mask


def _roles(model: Any, _context: Any) -> dict[str, Any] | None:
    text = model.model.language_model
    config = text.config
    actual = [layer.layer_type for layer in text.layers]
    actual_ple = [idx + 1 for idx, layer in enumerate(text.layers) if layer.ple is not None]
    if actual != config.layer_types or actual_ple != config.ple_layer_ids:
        return {"decoder_roles": actual, "ple_layers": actual_ple}
    if len(model.model.visual.blocks) != model.config.vision_config.depth or len(model.mtp.layers) != 1:
        return {"vision_blocks": len(model.model.visual.blocks), "draft_layers": len(model.mtp.layers)}
    return None


def get_validation_spec() -> ModelValidationSpec:
    """Declare parity for the backbone and draft plus runtime module coverage."""
    media = ("pixel_values", "image_grid_thw", "mm_token_type_ids")
    return ModelValidationSpec(
        module_cases=(
            ModuleParityCase(
                "gdn_reference_replacement", "model.language_model.layers.0.linear_attn",
                _candidate_gdn, _gdn_reference, _gdn_inputs, _copy_native,
                objective=lambda output: output.float().square().mean(),
                required_dtypes=("float32", "bfloat16"),
            ),
            ModuleParityCase(
                "native_multimodal_backbone", "model", _candidate, _reference_backbone,
                _backbone_inputs, _copy_native,
                objective=lambda output: output.last_hidden_state.float().square().mean(),
                required_dtypes=("float32", "bfloat16"),
            ),
            ModuleParityCase(
                "draft_fusion_qsa_moe", "mtp", _candidate, _reference_draft, _draft_inputs, _copy_draft,
                objective=lambda output: output.float().square().mean(),
                required_dtypes=("float32", "bfloat16"),
            ),
        ),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "attention_mask", "labels"),
            cp_replicated_forward_fields=("input_ids", "attention_mask", "labels", *media),
            modality_fields=media,
            modality_parameter_patterns=("model.visual.patch_embed.proj.weight", "model.visual.merger.*"),
            labels_are_shifted=False,
        ),
        state_invariants=(StateInvariantSpec("released_layer_roles", _roles, phase="structure"),),
        parameter_probes=tuple(ParameterProbeSpec(match) for match in (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.3.mlp.experts.gate_up_proj",
            "model.language_model.layers.3.attn_hyper_connection.input_mix_weight_down.weight",
            "model.visual.patch_embed.proj.weight",
            "model.visual.merger.linear_fc2.weight",
            "mtp.fc_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.mlp.experts.gate_up_proj",
        )),
        topology_constraints=(
            TopologyConstraint("tp_pp_one", lambda topo: topo["tp"] == topo.get("pp", 1) == 1,
                               "Flash-Next recipe requires TP1 and PP1"),
        ),
    )

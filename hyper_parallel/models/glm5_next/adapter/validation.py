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
"""Model-owned probes and fixed-input parity for the GLM training entry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.glm5_next import modeling_glm5_next as hf

from hyper_parallel.models.glm5_next.adapter.build import build_glm5_next, load_config
from hyper_parallel.models.glm5_next.modeling_glm5_next import DecoderInput, Glm5NextPackedKDA
from hyper_parallel.models.glm5_next.sparse_attention import (
    DocumentFragment, Glm5NextSparseAttention, SparseAttentionOutput,
)
from hyper_parallel.models.validation_spec import (
    CheckpointValidationSpec, DataValidationSpec, ModelValidationSpec, ModuleParityCase,
    ParameterProbeSpec, StateInvariantSpec, TopologyConstraint,
)


class _HfPackedKDA(hf.Glm5NextTextLinearAttention):
    """Evaluate each packed document using the unchanged HF recurrent model."""

    def forward(self, hidden_states: torch.Tensor, *, geometry: DecoderInput, cp: Any) -> torch.Tensor:
        if cp is not None:
            raise ValueError("HF KDA parity runs without CP")
        with patch.object(hf, "chunk_kimi_delta_attention", hf.recurrent_kimi_delta_attention):
            forward = super().forward
            parts = [forward(hidden_states[:, lo:hi])
                     for lo, hi in zip(geometry.offsets, geometry.offsets[1:])]
        return torch.cat(parts, dim=1)


class _HfNoPE(hf.Glm5NextTextAttention):
    """Use HF's unabsorbed attention implementation as the GLM NoPE oracle."""

    def forward(
        self, hidden_states: torch.Tensor, *, fragments: tuple[DocumentFragment, ...],
        local_start: int, cp: Any, indexer_mode: str,
    ) -> SparseAttentionOutput:
        if cp is not None or local_start != 0 or indexer_mode != "frozen" or len(fragments) != 1:
            raise ValueError("HF NoPE parity expects one unpadded document")
        valid = torch.ones_like(hidden_states[..., 0], dtype=torch.bool)
        output = super().forward(hidden_states, attention_mask=valid)[0]
        return SparseAttentionOutput(output, output.new_zeros(()), output.new_zeros((), dtype=torch.long))


def _config_path(context: Any) -> str:
    return context.manifest.model["builder"]["config_path"]


def _candidate(context: Any) -> nn.Module:
    return build_glm5_next(_config_path(context), indexer_mode="frozen")


def _reference_kda(context: Any) -> nn.Module:
    return _HfPackedKDA(load_config(_config_path(context)).text_config, 0)


def _reference_nope(context: Any) -> nn.Module:
    return _HfNoPE(load_config(_config_path(context)).text_config, 3)


def _copy_weights(reference: nn.Module, candidate: nn.Module, context: Any) -> None:
    dtype = getattr(torch, context.dtype)
    generator = torch.Generator().manual_seed(53)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.05, generator=generator)
    reference.to(device=context.device, dtype=dtype)
    candidate.to(device=context.device, dtype=dtype)
    candidate.load_state_dict(reference.state_dict())


def _parity_inputs(context: Any, *, packed: bool) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(117)
    length = 67
    config = load_config(_config_path(context)).text_config
    width = getattr(config, "hidden_size")
    value = torch.randn(1, length, width, generator=generator).to(context.device, getattr(torch, context.dtype))
    kwargs = ({"geometry": DecoderInput((0, 65, length), (), 0, length)}
              if packed else {"fragments": (DocumentFragment(0, length, 0),),
                              "local_start": 0, "indexer_mode": "frozen"})
    return {"args": (value.requires_grad_(),), "kwargs": dict(kwargs, cp=None)}


def _kda_inputs(context: Any) -> dict[str, Any]:
    return _parity_inputs(context, packed=True)


def _nope_inputs(context: Any) -> dict[str, Any]:
    return _parity_inputs(context, packed=False)


def _train_structure(model: nn.Module, context: Any) -> dict[str, Any] | None:
    del context
    layers = model.model.language_model.layers
    inventory = {
        "kda": sum(isinstance(layer.self_attn, Glm5NextPackedKDA) for layer in layers),
        "nope": sum(isinstance(layer.self_attn, Glm5NextSparseAttention) for layer in layers),
        "moe": sum(isinstance(layer.mlp, hf.Glm5NextTextMoE) for layer in layers),
        "text_layers": len(layers),
        "vision_layers": len(model.model.visual.blocks),
    }
    expected = {"kda": 34, "nope": 12, "moe": 43, "text_layers": 46, "vision_layers": 24}
    return None if inventory == expected else {"actual": inventory, "expected": expected}


def _trainer_topology(case: Mapping[str, Any]) -> bool:
    return case.get("tp", 1) == case.get("pp", 1) == 1 and case.get("cp", 1) in (1, 2)


def get_validation_spec() -> ModelValidationSpec:
    """Return structural, parity and end-to-end observables for GLM-5.3-Flash."""
    layers = "model.language_model.layers."
    vision = ("model.visual.patch_embed.proj.weight", "model.visual.merger.proj.weight")
    probes = (
        *vision, layers + "0.self_attn.q_proj.weight", layers + "3.self_attn.q_b_proj.weight",
        layers + "3.self_attn.indexer.index_kpool_compress_gate",
        layers + "3.mlp.experts.down_proj", layers + "45.eh_proj.weight", "lm_head.weight",
    )
    return ModelValidationSpec(
        module_cases=(
            ModuleParityCase("packed_kda", layers + "0.self_attn", _candidate, _reference_kda,
                             _kda_inputs, _copy_weights, required_parameter_patterns=("conv1d.weight",)),
            ModuleParityCase("absorbed_nope", layers + "3.self_attn", _candidate, _reference_nope,
                             _nope_inputs, _copy_weights, required_parameter_patterns=("kv_b_proj.weight",)),
        ),
        parameter_probes=tuple(ParameterProbeSpec(match) for match in probes),
        state_invariants=(StateInvariantSpec("released_crop_roles", _train_structure, phase="structure"),),
        checkpoint=CheckpointValidationSpec(),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "packed_doc_bounds", "shift_labels", "mtp_labels"),
            cp_replicated_forward_fields=("packed_doc_bounds", "pixel_values", "pixel_values_videos",
                                          "image_grid_thw", "video_grid_thw"),
            modality_fields=("pixel_values", "pixel_values_videos"),
            modality_parameter_patterns=vision,
            labels_are_shifted=True,
        ),
        topology_constraints=(TopologyConstraint("GLM CP head divisibility", _trainer_topology,
                                                  "Use TP1, PP1 and CP1/CP2 for the 2-head A5 crop."),),
    )

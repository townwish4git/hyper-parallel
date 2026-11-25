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
"""Training architecture for GLM-5.3-Flash, with packed and multimodal inputs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import nn
from transformers import Glm5NextConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.utils import ModelOutput
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    Glm5NextPreTrainedModel,
    Glm5NextTextDecoderLayer,
    Glm5NextTextHyperHead,
    Glm5NextTextLinearAttention,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
    Glm5NextVisionModel,
)

from hyper_parallel.components.losses.chunked_cross_entropy import chunked_cross_entropy
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
from hyper_parallel.components.ops import causal_conv1d, kda
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.glm5_next.sparse_attention import (
    DocumentFragment, Glm5NextSparseAttention, SparseAttentionOutput,
)


def document_bounds(length: int, boundaries: torch.Tensor | None) -> tuple[int, ...]:
    """Resolve validated token offsets with one host read per microbatch."""
    offsets = (0, length) if boundaries is None else tuple(boundaries.reshape(-1).tolist())
    if (len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != length
            or any(not isinstance(offset, int) for offset in offsets)
            or any(lo >= hi for lo, hi in zip(offsets, offsets[1:]))):
        raise ValueError("GLM packed boundaries must be strictly increasing and span the input row")
    return offsets


def future_positions(values: torch.Tensor, offsets: tuple[int, ...], distance: int, fill: int) -> torch.Tensor:
    """Align future tokens with their source position without crossing document boundaries."""
    aligned = torch.full_like(values, fill)
    for start, end in zip(offsets, offsets[1:]):
        if end - start > distance:
            aligned[:, start:end - distance] = values[:, start + distance:end]
    return aligned


@dataclass(frozen=True)
class DecoderInput:
    """Document resets and this rank's query interval after modality fusion."""

    offsets: tuple[int, ...]
    fragments: tuple[DocumentFragment, ...]
    start: int
    stop: int


def _decoder_input(offsets: tuple[int, ...], mask: torch.Tensor | None, start: int, stop: int) -> DecoderInput:
    """Treat padding as independent recurrence intervals and skip its sparse attention."""
    valid = []
    if mask is None:
        valid = list(zip(offsets, offsets[1:]))
    else:
        host_mask = mask.reshape(-1).tolist()
        for begin, end in zip(offsets, offsets[1:]):
            span = begin
            for position in range(begin, end):
                if not host_mask[position]:
                    if span < position:
                        valid.append((span, position))
                    span = position + 1
            if span < end:
                valid.append((span, end))
    fragments = tuple(
        DocumentFragment(max(start, begin), min(stop, end), begin)
        for begin, end in valid if begin < stop and end > start
    )
    resets = tuple(sorted({*offsets, *(edge for interval in valid for edge in interval)}))
    return DecoderInput(resets, fragments, start, stop)


class Glm5NextPackedKDA(Glm5NextTextLinearAttention):
    """GLM parameters and split short convolutions with HyperParallel packed KDA."""

    def forward(
        self, hidden_states: torch.Tensor, *, geometry: DecoderInput, cp: CPContext | None,
    ) -> torch.Tensor:
        """Reset the convolution and recurrence at each global document start."""
        forget = self.forget_gate
        decay = forget.f_b_proj(forget.f_a_proj(hidden_states))
        projected = tuple(tensor.unflatten(-1, (self.num_heads, self.head_dim)) for tensor in (
            self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states), decay,
        ))
        head_start, head_stop = 0, self.num_heads
        beta = self.b_proj(hidden_states).unsqueeze(-1)
        if cp is not None:
            if self.num_heads % cp.size != 0:
                raise ValueError("GLM KDA heads must divide evenly across the Ulysses CP group")
            heads = self.num_heads // cp.size
            head_start, head_stop = cp.rank * heads, (cp.rank + 1) * heads
            exchanged = cp.sequence_to_head(torch.cat((*projected, beta), dim=-1))
            *projected, beta = exchanged.split((self.head_dim,) * 4 + (1,), dim=-1)
        info = PackedSequenceInfo(geometry.offsets, 0, projected[0].shape[1])
        begin, end = head_start * self.head_dim, head_stop * self.head_dim
        streams = tuple(
            causal_conv1d(tokens.flatten(2).to(weight.dtype), weight[begin:end], None, info)
            .to(tokens.dtype).unflatten(-1, (head_stop - head_start, self.head_dim))
            for tokens, weight in zip(projected[:3], self.conv1d.weight.chunk(3, dim=0))
        )
        result = kda(
            *streams, projected[3], beta.squeeze(-1), a_log=forget.A_log[head_start:head_stop],
            dt_bias=forget.dt_bias[begin:end], lower_bound=forget.safe_gate_lower_bound, info=info,
        )
        if cp is not None:
            result = cp.head_to_sequence(result)
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).unflatten(-1, (self.num_heads, self.head_dim))
        return self.o_proj(self.o_norm(result, gate).flatten(2))


def _residual_update(
    connection: nn.Module, hidden: torch.Tensor, update: torch.Tensor,
    post: torch.Tensor, combine: torch.Tensor,
) -> torch.Tensor:
    combine_residual = getattr(connection, "combine_residual", None)
    if combine_residual is not None:
        return combine_residual(hidden, update, post, combine)
    return post.to(hidden.dtype).unsqueeze(-1) * update.unsqueeze(-2) + (
        combine.to(hidden.dtype).transpose(-1, -2) @ hidden
    )


class Glm5NextTrainingDecoderLayer(Glm5NextTextDecoderLayer):
    """One pretrained GLM layer with document-aware KDA or sparse attention."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Build one layer with the published norms and residual streams."""
        super().__init__(config, layer_idx)
        self.self_attn = (
            Glm5NextPackedKDA(config, layer_idx) if self.block_type == "linear_attention"
            else Glm5NextSparseAttention(config, layer_idx)
        )

    def forward(
        self, hidden: torch.Tensor, *, geometry: DecoderInput, cp: CPContext | None,
        indexer_mode: str,
    ) -> SparseAttentionOutput:
        """Run both four-stream residual sublayers and return this layer's KPool loss."""
        post, combine, attention = self.attn_hc(hidden)
        attention = self.input_layernorm(attention)
        if self.block_type == "linear_attention":
            attention = self.self_attn(attention, geometry=geometry, cp=cp)
            result = SparseAttentionOutput(attention, attention.new_zeros((), dtype=torch.float32),
                                           attention.new_zeros((), dtype=torch.long))
        else:
            result = self.self_attn(
                attention, fragments=geometry.fragments, local_start=geometry.start, cp=cp, indexer_mode=indexer_mode,
            )
        hidden = _residual_update(self.attn_hc, hidden, result.hidden, post, combine)
        post, combine, mlp_input = self.ffn_hc(hidden)
        update = self.mlp(self.post_attention_layernorm(mlp_input))
        hidden = _residual_update(self.ffn_hc, hidden, update, post, combine)
        return SparseAttentionOutput(hidden, result.indexer_loss, result.indexer_count)


class Glm5NextTrainingMTP(nn.Module):
    """Checkpoint layer 45 with ordinary residuals and the main model's shared vocabulary head."""

    def __init__(self, config: Any) -> None:
        """Build the separate MTP norms, next-token projection, attention and experts."""
        super().__init__()
        auxiliary = copy(config)
        auxiliary.indexer_types = [*config.indexer_types, "full"]
        width, eps = config.hidden_size, config.rms_norm_eps
        self.enorm = Glm5NextTextRMSNorm(width, eps)
        self.hnorm = Glm5NextTextRMSNorm(width, eps)
        self.eh_proj = nn.Linear(2 * width, width, bias=False)
        self.input_layernorm = Glm5NextTextRMSNorm(width, eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(width, eps)
        self.self_attn = Glm5NextSparseAttention(auxiliary, config.num_hidden_layers)
        self.mlp = Glm5NextTextMoE(config)
        self.shared_head = nn.Module()
        self.shared_head.norm = Glm5NextTextRMSNorm(width, eps)

    def forward(
        self, hidden: torch.Tensor, next_embedding: torch.Tensor, *, geometry: DecoderInput,
        cp: CPContext | None, indexer_mode: str,
    ) -> SparseAttentionOutput:
        """Fuse the next visual/text embedding and predict two tokens ahead."""
        hidden = self.eh_proj(torch.cat((self.enorm(next_embedding), self.hnorm(hidden)), dim=-1))
        result = self.self_attn(
            self.input_layernorm(hidden), fragments=geometry.fragments, local_start=geometry.start,
            cp=cp, indexer_mode=indexer_mode,
        )
        hidden = hidden + result.hidden
        hidden = self.shared_head.norm(hidden + self.mlp(self.post_attention_layernorm(hidden)))
        return SparseAttentionOutput(hidden, result.indexer_loss, result.indexer_count)


class _TrainingLanguageModel(Glm5NextPreTrainedModel):
    """Build the complete decoder and its MTP block before parallel replacements."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Glm5NextTrainingDecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.layers.append(Glm5NextTrainingMTP(config))
        self.norm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_head = Glm5NextTextHyperHead()
        self.post_init()

    def forward(
        self, embeddings: torch.Tensor, next_embedding: torch.Tensor, *,
        geometry: DecoderInput, cp: CPContext | None, indexer_mode: str,
    ) -> tuple[torch.Tensor, SparseAttentionOutput]:
        hidden = embeddings.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        loss = hidden.new_zeros((), dtype=torch.float32)
        count = hidden.new_zeros((), dtype=torch.long)
        for layer in self.layers[:-1]:
            result = layer(hidden, geometry=geometry, cp=cp, indexer_mode=indexer_mode)
            hidden = result.hidden
            loss = loss + result.indexer_loss
            count = count + result.indexer_count
        main = self.norm(self.hc_head(hidden))
        mtp = self.layers[-1](main, next_embedding, geometry=geometry, cp=cp, indexer_mode=indexer_mode)
        return main, SparseAttentionOutput(mtp.hidden, loss + mtp.indexer_loss, count + mtp.indexer_count)


class _TrainingMultimodalModel(Glm5NextModel):
    """Keep the upstream media getters and placeholder conventions."""

    def __init__(self, config: Glm5NextConfig) -> None:
        # Reuse HF media and checkpoint methods with the complete training module tree.
        Glm5NextPreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        self.visual = Glm5NextVisionModel._from_config(config.vision_config)  # pylint: disable=protected-access
        rotary = self.visual.rotary_pos_emb
        with torch.device("cpu"):
            inv_freq, _ = rotary.compute_axial_rope_parameters(config.vision_config)
        for name in ("inv_freq", "original_inv_freq"):
            register_rebuildable_buffer(rotary, name, value=inv_freq)
        self.language_model = _TrainingLanguageModel(config.text_config)
        self.post_init()


@dataclass
class Glm5NextTrainingOutput(ModelOutput):
    """Named globally token-weightable objectives and detached accounting."""

    loss: dict[str, torch.Tensor] | None = None
    logits: torch.Tensor | None = None
    extra_metrics: dict[str, torch.Tensor] | None = None


class Glm5NextTrainingModel(Glm5NextForConditionalGeneration):
    """Training-only facade. Architecture and checkpoint names match the published GLM layout."""

    _keys_to_ignore_on_load_unexpected = []
    _no_split_modules = ["Glm5NextTrainingDecoderLayer", "Glm5NextTrainingMTP", "Glm5NextVisionBlock"]

    def __init__(self, config: Glm5NextConfig, cp: CPContext | None = None) -> None:
        """Construct the full training model and configure indexer gradient ownership."""
        # Reuse HF media and checkpoint methods with the complete training module tree.
        Glm5NextPreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        text = config.text_config
        if text.num_nextn_predict_layers != 1:
            raise ValueError("GLM-5.3-Flash training expects one additional prediction layer")
        self.cp = cp
        self.model = _TrainingMultimodalModel(config)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.indexer_mode = getattr(config, "hp_indexer_mode", "frozen")
        self.mtp_weight = getattr(config, "hp_mtp_weight", 0.1)
        self.kpool_weight = getattr(config, "hp_kpool_weight", 1.0)
        self.ce_chunk_size = getattr(config, "hp_ce_chunk_size", 256)
        if self.indexer_mode not in ("frozen", "dense_warmup", "sparse"):
            raise ValueError("GLM indexer_mode must be frozen, dense_warmup or sparse")
        if any(not isfinite(weight) or weight < 0 for weight in (self.mtp_weight, self.kpool_weight)):
            raise ValueError("GLM auxiliary loss weights must be finite and nonnegative")
        if not isinstance(self.ce_chunk_size, int) or self.ce_chunk_size < 1:
            raise ValueError("GLM ce_chunk_size must be a positive integer")
        self.post_init()
        for name, parameter in self.named_parameters():
            if ".self_attn.indexer." in name:
                parameter.requires_grad_(self.indexer_mode != "frozen")
            elif self.indexer_mode == "dense_warmup":
                parameter.requires_grad_(False)

    @classmethod
    def from_config(cls, config: Glm5NextConfig, **kwargs: Any) -> Glm5NextTrainingModel:
        """Construct through HyperAutoModel's custom architecture entry point."""
        return cls._from_config(config, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        """Return the published text embedding."""
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the input embedding without changing its checkpoint name."""
        self.model.language_model.embed_tokens = value

    def _embed(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None, image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None, video_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        embeddings = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            features = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            values = torch.cat(features.pooler_output).to(embeddings)
            mask, _ = self.model.get_placeholder_mask(input_ids, embeddings, image_features=values)
            embeddings = embeddings.masked_scatter(mask, values)
        if pixel_values_videos is not None:
            features = self.model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            values = torch.cat(features.pooler_output).to(embeddings)
            _, mask = self.model.get_placeholder_mask(input_ids, embeddings, video_features=values)
            embeddings = embeddings.masked_scatter(mask, values)
        return embeddings

    def _losses(
        self, hidden: torch.Tensor, mtp: SparseAttentionOutput, shift_labels: torch.Tensor,
        mtp_labels: torch.Tensor, normalizer: torch.Tensor,
    ) -> Glm5NextTrainingOutput:
        if self.indexer_mode == "dense_warmup":
            main_sum = mtp.indexer_loss.detach().new_zeros(())
            mtp_sum = main_sum
        else:
            main_sum = chunked_cross_entropy(hidden, shift_labels, self.lm_head.weight, self.ce_chunk_size)
            mtp_sum = chunked_cross_entropy(mtp.hidden, mtp_labels, self.lm_head.weight, self.ce_chunk_size)
        losses = {
            "foundation_loss": main_sum / normalizer,
            "foundation_loss_mtp": mtp_sum * self.mtp_weight / normalizer,
            "foundation_loss_kpool": mtp.indexer_loss * self.kpool_weight / normalizer,
        }
        statistics = {
            "glm_lm_tokens": (shift_labels != IGNORE_INDEX).sum().float(),
            "glm_mtp_tokens": (mtp_labels != IGNORE_INDEX).sum().float(),
            "glm_kpool_queries": mtp.indexer_count.float(),
        }
        return Glm5NextTrainingOutput(loss=losses, extra_metrics=statistics)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None, shift_labels: torch.Tensor | None = None,
        mtp_labels: torch.Tensor | None = None, packed_doc_bounds: torch.Tensor | None = None,
        loss_normalizer: torch.Tensor | None = None, pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None, pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None, use_cache: bool = False, **kwargs: Any,
    ) -> Glm5NextTrainingOutput:
        """Fuse full media embeddings before handing contiguous CP shards to the decoder."""
        del kwargs
        if use_cache or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("GLM training requires one packed input row and use_cache=False")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("GLM attention_mask must have the same shape as input_ids")
        if (shift_labels is None) != (mtp_labels is None):
            raise ValueError("GLM requires both pre-shifted targets when either is provided")
        if labels is not None and shift_labels is None and self.cp is not None:
            raise ValueError("CP training requires globally shifted labels from Glm5NextBatch")
        start, stop = 0, input_ids.shape[1]
        if self.cp is not None:
            fields = input_ids if attention_mask is None else torch.cat((input_ids, attention_mask.to(input_ids)), 0)
            fields, partition = self.cp.gather_sequence(fields)
            start = partition.offset
            stop = start + input_ids.shape[1]
            input_ids = fields[:1]
            if attention_mask is not None:
                attention_mask = fields[1:].bool()
        offsets = document_bounds(input_ids.shape[1], packed_doc_bounds)
        geometry = _decoder_input(offsets, attention_mask, start, stop)
        embeddings = self._embed(input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        future = future_positions(embeddings, geometry.offsets, 1, 0)
        hidden, mtp = self.model.language_model(
            embeddings[:, start:stop], future[:, start:stop], geometry=geometry, cp=self.cp,
            indexer_mode=self.indexer_mode,
        )
        if labels is None and shift_labels is None:
            return Glm5NextTrainingOutput(logits=self.lm_head(hidden))
        if shift_labels is None:
            if attention_mask is not None:
                labels = labels.masked_fill(~attention_mask.bool(), IGNORE_INDEX)
            shift_labels = future_positions(labels, geometry.offsets, 1, IGNORE_INDEX)
            mtp_labels = future_positions(labels, geometry.offsets, 2, IGNORE_INDEX)
        denominator = loss_normalizer
        if denominator is None:
            denominator = (input_ids.new_tensor(stop - start) if attention_mask is None
                           else attention_mask[:, start:stop].bool().sum())
        return self._losses(hidden, mtp, shift_labels, mtp_labels, denominator.clamp_min(1))


# Transformers applies conversions to custom architectures only when registered explicitly.
register_checkpoint_conversion_mapping(
    Glm5NextTrainingModel.__name__, get_checkpoint_conversion_mapping("glm5_next"), overwrite=True,
)


def construct_model(
    config: Glm5NextConfig, *, context: Mapping[str, Any] | None = None,
) -> Glm5NextTrainingModel:
    """Bind model-owned CP transport to HyperParallel's existing mesh."""
    mesh = (context or {}).get("mesh")
    cp = CPContext(mesh.cp_mesh) if mesh is not None and mesh.cp_size > 1 else None
    return Glm5NextTrainingModel(config, cp=cp)

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
"""GLM adapters for optimized vision, residual mixing and gated FFN components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionAttention

from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.components.modules.mhc.pipelined_mhc import PipelinedMhcModule
from hyper_parallel.components.modules.swiglu_mlp_family import swiglu_activation
from hyper_parallel.components.ops import deepseek_sparse_attention_absorbed, mhc_post, rope_qk, vision_attention
from hyper_parallel.models.glm5_next.sparse_attention import Glm5NextSparseAttention
from hyper_parallel.models.replacement import module_replacement


class Glm5NextAlignedSparseAttention(Glm5NextSparseAttention):
    """Prepare indices for the A5 kernel contract and native shared-bank path."""

    _attention_kernel = staticmethod(deepseek_sparse_attention_absorbed)

    @staticmethod
    def _kernel_indices(indices: torch.Tensor, key_len: int) -> torch.Tensor:
        # Keep all selected tokens, including the tail, ahead of the sentinels.
        ordered = indices.masked_fill(indices < 0, key_len).sort(-1).values
        ordered = ordered.masked_fill(ordered == key_len, -1)
        return F.pad(ordered, (0, -ordered.shape[-1] % 1024), value=-1)


class GlmVisionAttention(Glm5NextVisionAttention):
    """Dispatch each image/video-frame segment to HyperParallel vision attention."""

    def __init__(self, module: Glm5NextVisionAttention) -> None:
        """Keep the source projection and normalization parameters."""
        super().__init__(module.config)
        self.qkv = module.qkv
        self.proj = module.proj
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm
        if self.attention_dropout:
            raise ValueError("GLM optimized vision attention requires zero dropout")

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None, **kwargs: Any,
    ) -> torch.Tensor:
        """Preserve HF two-dimensional rotary embeddings and per-frame attention."""
        del max_seqlen, kwargs
        if position_embeddings is None:
            raise ValueError("GLM vision attention requires two-dimensional rotary embeddings")
        count = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).view(count, 3, self.num_heads, self.head_dim).unbind(1)
        query, key = self.q_norm(query), self.k_norm(key)
        cos, sin = position_embeddings
        query, key = rope_qk(query.float().unsqueeze(0), key.float().unsqueeze(0),
                             cos, sin, unsqueeze_dim=2)
        query, key = query.to(value.dtype).transpose(1, 2), key.to(value.dtype).transpose(1, 2)
        value = value.transpose(0, 1).unsqueeze(0)
        offsets = cu_seqlens.tolist()
        mixed = torch.cat([
            vision_attention(query[:, :, lo:hi], key[:, :, lo:hi], value[:, :, lo:hi])
            for lo, hi in zip(offsets, offsets[1:])
        ], dim=1)
        return self.proj(mixed.reshape(count, -1))


class GlmMhc(PipelinedMhcModule):
    """Keep released residual parameters and choose the pre-kernel by hidden width."""

    def reset_parameters(self) -> None:
        """Initialize the adopted GLM parameters after meta materialization."""
        nn.init.normal_(self.fn, std=0.02)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return HF's post/combination coefficients and collapsed hidden stream."""
        width = hidden_streams.shape[-1]
        if hidden_streams.device.type == "npu" and width in (4096, 7168) and hidden_streams.dtype == torch.bfloat16:
            collapsed, post, combine, _ = mhc_pre(
                hidden_streams.flatten(2), self.fn, self.scale, self.base, self.hc_mult,
                self.hc_sinkhorn_iters, self.input_norm.eps, self.hc_eps,
            )
            return post, combine, collapsed
        pre, post, combine = super().forward(hidden_streams)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, combine, collapsed

    def combine_residual(
        self, hidden: torch.Tensor, update: torch.Tensor, post: torch.Tensor, combine: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the matching post-kernel to the original four streams."""
        return mhc_post(update, hidden.flatten(2), post, combine, self.hc_mult).reshape_as(hidden)


class GlmClampedMLP(nn.Module):
    """Reuse the published separate projections and HyperParallel's clamped SwiGLU."""

    def __init__(self, module: nn.Module) -> None:
        """Keep the source gate, up and down projections."""
        super().__init__()
        self.gate_proj = module.gate_proj
        self.up_proj = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn = module.act_fn
        self.limit = module.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the source gate/up clamps and project down."""
        gate, up = self.gate_proj(x), self.up_proj(x)
        activated = swiglu_activation(gate, up, limit=self.limit, activation_dtype="float32")
        return self.down_proj(activated.to(gate.dtype))


@module_replacement
def replace_vision(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> nn.Module:
    """Replace vision compute while preserving the QKV and output parameters."""
    del module_fqn, context
    return GlmVisionAttention(module)


@module_replacement
def replace_mhc(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> nn.Module:
    """Choose the portable or fused MHC pre path at runtime."""
    del module_fqn, context
    return GlmMhc(module=module)


@module_replacement
def replace_ffn(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> nn.Module:
    """Select the GLM clamped SwiGLU implementation for ordinary and shared FFNs."""
    del module_fqn, context
    return GlmClampedMLP(module)

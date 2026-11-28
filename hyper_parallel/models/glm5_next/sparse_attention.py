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
"""GLM KPool selection and NoPE attention over packed document prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.nn import functional as F
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

from hyper_parallel.components.ops import DsaQueryLayout, dsa_indexer, native_deepseek_sparse_attention_absorbed


@dataclass(frozen=True)
class DocumentFragment:
    """Queries and the corresponding prefix in document-aligned token coordinates."""

    start: int
    stop: int
    document_start: int


class SparseAttentionOutput(NamedTuple):
    """Local projected outputs and the sum and count of supervised KPool queries."""

    hidden: torch.Tensor
    indexer_loss: torch.Tensor
    indexer_count: torch.Tensor


def _pool_keys(keys: torch.Tensor, gates: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """Reduce complete pools with per-channel softmax weights."""
    pool_size, width = ape.shape
    length = keys.shape[1] // pool_size * pool_size
    key_blocks = keys[:, :length].reshape(keys.shape[0], -1, pool_size, width)
    logits = gates[:, :length].float().reshape_as(key_blocks) + ape.float()
    return (logits.softmax(dim=2).to(keys.dtype) * key_blocks).sum(dim=2)


def _expand_pool_ids(pool_ids: torch.Tensor, positions: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Expand complete pools and append up to pool_size - 1 visible tail tokens."""
    offsets = torch.arange(pool_size, dtype=torch.int32, device=pool_ids.device)
    raw = pool_ids.unsqueeze(-1) * pool_size + offsets
    raw = raw.masked_fill(pool_ids.unsqueeze(-1) < 0, -1).flatten(-2)
    visible = positions + 1
    tail_count = visible % pool_size
    tail_offset = offsets[:-1]
    tail = (visible - tail_count).unsqueeze(-1) + tail_offset
    tail = tail.masked_fill(tail_offset >= tail_count.unsqueeze(-1), -1)
    return torch.cat((raw, tail.unsqueeze(0)), dim=-1)


def _pool_kl(
    query: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor,
    teacher_query: torch.Tensor, teacher_key: torch.Tensor, positions: torch.Tensor,
    pool_size: int, scale: float, selected: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill detached main-attention probability mass over complete causal pools."""
    count = keys.shape[1]
    eligible = (positions + 1) // pool_size
    supervised = eligible >= 2 if selected is None else (selected >= 0).sum(-1).squeeze(0) >= 2
    if count < 2:
        return (query.sum(dtype=torch.float32) + keys.sum(dtype=torch.float32)
                + weights.sum(dtype=torch.float32)) * 0.0, supervised.sum()

    total = query.new_zeros((), dtype=torch.float32)
    pool_ids = torch.arange(count, device=query.device)
    for start in range(0, query.shape[1], 64):
        stop = min(start + 64, query.shape[1])
        visible = pool_ids < eligible[start:stop, None]
        if selected is not None:
            pool_ids_in_chunk = selected[0, start:stop].long()
            selected_slots = torch.zeros_like(visible, dtype=torch.int32)
            selected_slots.scatter_add_(-1, pool_ids_in_chunk.clamp_min(0), (pool_ids_in_chunk >= 0).int())
            visible = visible & selected_slots.bool()
        usable = supervised[start:stop]
        # A finite dummy distribution keeps unsupervised causal-prefix rows differentiable.
        safe = torch.where(usable[:, None], visible, pool_ids == 0)
        logits = torch.einsum("bqhd,bpd->bqhp", query[:, start:stop].float(), keys.float()).relu()
        logits = (logits * weights[:, start:stop, :, None]).sum(dim=2)
        student = logits.masked_fill(~safe, torch.finfo(logits.dtype).min).log_softmax(dim=-1)
        with torch.no_grad():
            scores = torch.einsum(
                "bqhd,bkd->bqhk", teacher_query[:, start:stop].float(), teacher_key[:, :count * pool_size].float(),
            )
            scores = (scores * scale).reshape(1, stop - start, scores.shape[2], count, pool_size)
            scores = scores.masked_fill(~safe[None, :, None, :, None], -torch.inf)
            target = scores.flatten(-2).softmax(dim=-1).reshape_as(scores).sum(dim=-1).mean(dim=2)
        per_query = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - student)).sum(-1)
        total = total + (per_query * usable).sum()
    return total, supervised.sum()


class Glm5NextSparseAttention(Glm5NextTextAttention):
    """Preserve HF projection names and select against raw document-relative KV."""

    _attention_kernel = staticmethod(native_deepseek_sparse_attention_absorbed)

    def __init__(self, config: Glm5NextTextConfig, layer_idx: int) -> None:
        """Build the NoPE projections and the per-layer KPool indexer."""
        super().__init__(config, layer_idx)
        if (self.indexer is None or self.qk_rope_head_dim != 0 or self.attention_dropout
                or not self.indexer.index_kpool_always_select_tail):
            raise ValueError("GLM training expects NoPE, a full KPool indexer with tail selection, and zero dropout")

    def _project_query(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.q_a_layernorm(self.q_a_proj(hidden))
        query = self.q_b_proj(residual).unflatten(-1, (self.num_heads, self.qk_nope_head_dim))
        projection = self.kv_b_proj.weight.reshape(self.num_heads, -1, self.kv_lora_rank)
        key_weight, value_weight = projection.split((self.qk_nope_head_dim, self.v_head_dim), dim=1)
        return residual, torch.einsum("bqhd,hdk->bqhk", query, key_weight), value_weight

    def _index_states(self, hidden: torch.Tensor, residual: torch.Tensor, train_indexer: bool) -> tuple:
        # The indexer learns its projections and pool parameters without shaping the teacher or decoder input.
        indexer = self.indexer
        with torch.set_grad_enabled(torch.is_grad_enabled() and train_indexer):
            inputs = hidden.detach()
            query = indexer.wq_b(residual.detach()).unflatten(-1, (indexer.n_heads, indexer.head_dim))
            keys = indexer.k_norm(indexer.wk(inputs))
            gates = F.linear(inputs, indexer.index_kpool_compress_gate)
            weights = indexer.weights_proj(inputs).float() * (indexer.n_heads ** -0.5 * indexer.softmax_scale)
        return query, keys, gates, weights

    def _attend_prefix(
        self, query: torch.Tensor, kv: torch.Tensor, query_index: torch.Tensor, pool_keys: torch.Tensor,
        index_weight: torch.Tensor, indexer_mode: str,
    ) -> SparseAttentionOutput:
        query_len, key_len = query.shape[1], kv.shape[1]
        ratio = self.indexer.index_kpool
        positions = torch.arange(key_len - query_len, key_len, dtype=torch.int32, device=query.device)
        with torch.no_grad():
            layout = DsaQueryLayout(
                compress_ratio=ratio, segments=((0, query_len, 0, key_len // ratio, key_len % ratio),),
            )
            if indexer_mode == "dense_warmup":
                candidates = torch.arange(key_len // ratio, device=query.device, dtype=torch.int32)
                selected = candidates.expand(1, query_len, -1).masked_fill(
                    candidates >= (positions[:, None] + 1) // ratio, -1,
                )
            else:
                selected = dsa_indexer(query_index, pool_keys, index_weight, layout=layout,
                                       sparse_count=self.indexer.index_topk // ratio)
            indices = _expand_pool_ids(selected, positions, ratio).squeeze(0).unsqueeze(1)
            indices = self._kernel_indices(indices, key_len)
            query_ends = positions.new_tensor([query_len])
            key_ends = positions.new_tensor([key_len])
        q_rope = query.new_zeros((*query.shape[:-1], 64))
        k_rope = kv.new_zeros((*kv.shape[:2], 1, 64))
        out = self._attention_kernel(
            query, kv.unsqueeze(2), q_rope, k_rope, indices, self.scaling, query_ends, key_ends,
        )[0][..., :self.kv_lora_rank]
        if indexer_mode != "frozen":
            supervised = None if indexer_mode == "dense_warmup" else selected
            loss, count = _pool_kl(
                query_index, pool_keys, index_weight, query, kv, positions, ratio, self.scaling, supervised,
            )
        else:
            loss = query.new_zeros((), dtype=torch.float32)
            count = positions.new_zeros(())
        return SparseAttentionOutput(out, loss, count)

    @staticmethod
    def _kernel_indices(indices: torch.Tensor, key_len: int) -> torch.Tensor:
        """Pass the selected token positions to the portable kernel."""
        del key_len
        return indices

    def _prefix_bank(self, bank: torch.Tensor) -> torch.Tensor:
        """Return the projected inputs visible to the document queries."""
        return bank

    def forward(
        self, hidden_states: torch.Tensor, *, fragments: tuple[DocumentFragment, ...],
        local_start: int, indexer_mode: str,
    ) -> SparseAttentionOutput:
        """Run document-local attention over the available prefixes."""
        residual, query, value_weight = self._project_query(hidden_states)
        kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        index_query, index_key, gates, weights = self._index_states(hidden_states, residual, indexer_mode != "frozen")
        bank = self._prefix_bank(torch.cat((kv, index_key, gates), dim=-1))
        key_bank, index_bank, gate_bank = bank.split(
            (self.kv_lora_rank, self.indexer.head_dim, self.indexer.head_dim), dim=-1,
        )
        # Empty query windows still keep the projected inputs in the backward graph.
        anchor = bank[:, :0].sum()
        loss = anchor.float() + index_query[:, :0].sum(dtype=torch.float32) + weights[:, :0].sum()
        if indexer_mode != "frozen":
            loss = loss + self.indexer.index_kpool_compress_ape[:0].sum(dtype=torch.float32)
        count = torch.zeros((), dtype=torch.long, device=hidden_states.device)
        local = torch.zeros_like(query) + anchor + query[:, :0].sum()
        for fragment in fragments:
            source = slice(fragment.document_start, fragment.stop)
            target = slice(fragment.start - local_start, fragment.stop - local_start)
            pool_keys = _pool_keys(index_bank[:, source], gate_bank[:, source], self.indexer.index_kpool_compress_ape)
            part = self._attend_prefix(
                query[:, target], key_bank[:, source], index_query[:, target], pool_keys, weights[:, target],
                indexer_mode,
            )
            local[:, target] = part.hidden
            loss = loss + part.indexer_loss
            count = count + part.indexer_count
        output = torch.einsum("bqhk,hvk->bqhv", local, value_weight).flatten(2)
        return SparseAttentionOutput(self.o_proj(output), loss, count)

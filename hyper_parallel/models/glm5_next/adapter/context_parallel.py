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
"""Context-parallel variants of the standalone GLM training model."""

from __future__ import annotations

from functools import partial

import torch
from transformers import Glm5NextConfig
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.models.glm5_next.adapter.replacements import Glm5NextAlignedSparseAttention
from hyper_parallel.models.glm5_next.modeling_glm5_next import (
    Glm5NextPackedKDA, Glm5NextTrainingModel, build_text_attention,
)


class Glm5NextCPKDA(Glm5NextPackedKDA):
    """Apply the same packed recurrence after an Ulysses head exchange."""

    def __init__(self, config: Glm5NextTextConfig, layer_idx: int, *, cp: CPContext) -> None:
        """Bind the CP mesh and determine the local head interval."""
        super().__init__(config, layer_idx)
        if self.num_heads % cp.size:
            raise ValueError("GLM KDA heads must divide evenly across the Ulysses CP group")
        self.cp = cp
        heads = self.num_heads // cp.size
        self.head_start = cp.rank * heads
        self.head_stop = self.head_start + heads

    def _run_recurrence(
        self, projected: tuple[torch.Tensor, ...], beta: torch.Tensor, offsets: tuple[int, ...],
    ) -> torch.Tensor:
        packed = self.cp.sequence_to_head(torch.cat((*projected, beta), dim=-1))
        *head_shards, head_beta = packed.split((self.head_dim,) * 4 + (1,), dim=-1)
        result = self._apply_recurrence(tuple(head_shards), head_beta, offsets, self.head_start, self.head_stop)
        return self.cp.head_to_sequence(result)


class Glm5NextCPSparseAttention(Glm5NextAlignedSparseAttention):
    """Gather the raw KV and pool inputs before local NoPE attention."""

    def __init__(self, config: Glm5NextTextConfig, layer_idx: int, *, cp: CPContext) -> None:
        """Bind the differentiable sequence gather."""
        super().__init__(config, layer_idx)
        self.cp = cp

    def _prefix_bank(self, bank: torch.Tensor) -> torch.Tensor:
        gathered, _ = self.cp.gather_sequence(bank)
        return gathered


class Glm5NextCPTrainingModel(Glm5NextTrainingModel):
    """Fuse globally aligned media inputs and run the decoder on local tokens."""

    def __init__(self, config: Glm5NextConfig, *, cp: CPContext) -> None:
        """Construct the decoder and MTP with the CP attention variants."""
        attention_factory = partial(
            build_text_attention, linear=partial(Glm5NextCPKDA, cp=cp),
            sparse=partial(Glm5NextCPSparseAttention, cp=cp),
        )
        super().__init__(config, attention_factory=attention_factory)
        self.cp = cp

    def _prepare_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, *,
        labels: torch.Tensor | None, shift_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        if labels is not None and shift_labels is None:
            raise ValueError("CP training requires targets aligned before sharding by Glm5NextRuntimeInputs")
        fields = input_ids if attention_mask is None else torch.cat((input_ids, attention_mask.to(input_ids)), dim=0)
        gathered, partition = self.cp.gather_sequence(fields)
        global_mask = gathered[1:].bool() if attention_mask is not None else None
        start = partition.offset
        return gathered[:1], global_mask, start, start + input_ids.shape[1]

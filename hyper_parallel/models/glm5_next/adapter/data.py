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
"""GLM batching for globally shifted targets across contiguous CP shards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.glm5_next.modeling_glm5_next import document_bounds, future_positions


class Glm5NextRuntimeInputs(RuntimeInputAdapter):
    """Build t+1/t+2 targets before splitting text, leaving media fields global."""

    def cp_token_fields(self) -> tuple[str, ...]:
        """Shard precomputed targets alongside the decoder tokens."""
        return ("shift_labels", "mtp_labels")

    def prepare_source_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Prepare packed prediction targets from the unsharded batch."""
        labels = batch["labels"]
        mask = batch.get("attention_mask")
        valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else mask.bool()
        labels = labels.masked_fill(~valid | ~batch["loss_mask"].bool(), IGNORE_INDEX)
        offsets = document_bounds(labels.shape[1], batch.get("cu_seq_lens"))
        shift_labels = future_positions(labels, offsets, 1, IGNORE_INDEX).masked_fill(~valid, IGNORE_INDEX)
        mtp_labels = future_positions(labels, offsets, 2, IGNORE_INDEX)
        next_valid = future_positions(valid, offsets, 1, 0)
        return {
            **batch,
            "attention_mask": valid,
            "cu_seq_lens": torch.tensor(offsets, dtype=torch.int32, device=labels.device),
            "shift_labels": shift_labels,
            "mtp_labels": mtp_labels.masked_fill(~(valid & next_valid), IGNORE_INDEX),
        }

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the forward-only inputs derived after transport."""
        return ("packed_doc_bounds", "loss_normalizer")

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Forward the global boundaries and the valid local input-token count."""
        del context
        return {"packed_doc_bounds": batch["cu_seq_lens"], "loss_normalizer": batch["attention_mask"].sum()}

    def prepare_loss_inputs(self, *, batch: Mapping[str, Any], loss_inputs: dict[str, Any]) -> dict[str, Any]:
        """Use the same valid input-token budget for Trainer loss weighting."""
        valid = batch["attention_mask"]
        # All objectives are sums over the same global packed input-token budget.
        # Prompt tokens can train KPool even on a shard with no supervised LM labels.
        return {
            **loss_inputs,
            "shift_labels": torch.zeros_like(batch["shift_labels"]).masked_fill(~valid, IGNORE_INDEX),
            "loss_mask": valid,
        }

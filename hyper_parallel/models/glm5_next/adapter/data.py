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

from hyper_parallel.data.batching import OmniParallelBatch
from hyper_parallel.data.constants import IGNORE_INDEX, OMNI_CP_TOKEN_FIELDS
from hyper_parallel.data.parallel.batch_parallel import CPBatchSharder
from hyper_parallel.models.glm5_next.modeling_glm5_next import document_bounds, future_positions


class Glm5NextBatch(OmniParallelBatch):
    """Build t+1/t+2 targets before splitting text, leaving media fields global."""

    def __init__(self, **kwargs: Any) -> None:
        """Shard precomputed prediction targets alongside the decoder tokens."""
        super().__init__(**kwargs)
        self.cp_sharder = CPBatchSharder(
            self.parallel_context, token_fields=OMNI_CP_TOKEN_FIELDS.union(("shift_labels", "mtp_labels")),
        )

    @staticmethod
    def _normalize_source_batch(source_batch: Mapping[str, Any]) -> dict[str, Any]:
        batch = OmniParallelBatch._normalize_source_batch(source_batch)
        labels = batch["labels"]
        valid = batch.get("attention_mask", torch.ones_like(labels)).bool()
        labels = labels.masked_fill(~valid | ~batch["loss_mask"].bool(), IGNORE_INDEX)
        bounds = batch.get("cu_seq_lens")
        offsets = document_bounds(labels.shape[1], bounds)
        batch["attention_mask"] = valid
        batch["packed_doc_bounds"] = torch.tensor(offsets, dtype=torch.int32, device=labels.device)
        batch["shift_labels"] = future_positions(labels, offsets, 1, IGNORE_INDEX).masked_fill(~valid, IGNORE_INDEX)
        mtp_labels = future_positions(labels, offsets, 2, IGNORE_INDEX)
        next_valid = future_positions(valid, offsets, 1, 0)
        batch["mtp_labels"] = mtp_labels.masked_fill(~(valid & next_valid), IGNORE_INDEX)
        return batch

    @staticmethod
    def _split_model_and_loss_inputs(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        model_inputs, loss_inputs = OmniParallelBatch._split_model_and_loss_inputs(batch)
        valid = batch["attention_mask"].bool()
        # All objectives are sums over the same global packed input-token budget.
        # Prompt tokens can train KPool even on a shard with no supervised LM labels.
        loss_inputs["shift_labels"] = torch.zeros_like(batch["shift_labels"]).masked_fill(~valid, IGNORE_INDEX)
        loss_inputs["loss_mask"] = valid
        model_inputs["loss_normalizer"] = valid.sum()
        return model_inputs, loss_inputs

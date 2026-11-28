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
"""Global media inputs for the Flash-Next decoder's internal context split."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.batching.get_batch import OmniParallelBatch


class FlashNextBatch(OmniParallelBatch):
    """Keep media insertion, RoPE metadata and loss targets global inside each CP group."""

    labels_are_shifted = False

    def __call__(
        self, data_iterator: Any, *, external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Move the unsliced batch to the device and preserve the model/loss field contract."""
        source = external_batch if external_batch is not None else next(data_iterator)
        normalized = self._normalize_source_batch(source)
        if normalized["input_ids"].shape[1] % self.parallel_context.cp_world_size:
            raise ValueError("Flash-Next global token length must be divisible by CP size")
        if normalized.get("cu_seq_lens") is not None:
            raise ValueError("Flash-Next training requires unpacked examples")
        batch = self.tp_broadcaster.broadcast(normalized, broadcast_all_fields=True)
        inputs, loss_inputs = self._split_model_and_loss_inputs(batch)
        inputs.update(self._build_runtime_inputs(batch))
        return inputs, loss_inputs

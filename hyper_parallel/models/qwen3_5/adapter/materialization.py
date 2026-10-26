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
"""Restore qwen3_5 rotary buffers after meta-device materialization."""

# Native HF rotary modules and their buffer lifecycle use the PyTorch contract.
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
)

from hyper_parallel.models.materialization import register_rebuildable_buffer


def register_materialization(model: nn.Module) -> None:
    """Capture construction-time RoPE values before sharding and storage allocation.

    Args:
        model: Constructed native model whose rotary buffers are still initialized.
    """
    for module in model.modules():
        if isinstance(module, Qwen3_5TextRotaryEmbedding):
            register_rebuildable_buffer(module, "inv_freq")
            register_rebuildable_buffer(module, "original_inv_freq")
        elif isinstance(module, Qwen3_5VisionRotaryEmbedding):
            register_rebuildable_buffer(module, "inv_freq")

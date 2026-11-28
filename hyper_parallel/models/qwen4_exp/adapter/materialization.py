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
"""Rebuild Flash-Next's derived text, MTP and vision RoPE buffers."""

import torch
from torch import nn
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpVisionRotaryEmbedding,
)

from hyper_parallel.models.materialization import register_rebuildable_buffer


def register_rotary_buffers(model: nn.Module) -> None:
    """Record deterministic RoPE state outside the checkpoint-owned N-gram buffers."""
    for module in model.modules():
        if isinstance(module, (Qwen4ExpTextRotaryEmbedding, Qwen4ExpVisionRotaryEmbedding)):
            with torch.device("cpu"):
                reference = type(module)(module.config)
            for name, value in reference.named_buffers(recurse=False):
                register_rebuildable_buffer(module, name, value=value)

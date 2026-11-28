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
"""Depth-preserving random-init Flash-Next configuration for training validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

from hyper_parallel.models import HyperAutoModelForImageTextToText


def cropped_config(config_path: str, *, sparse_budget: int | None = None) -> Qwen4ExpConfig:
    """Keep released layer roles, shrink widths and validate the expert/top-k constraints."""
    if not (Path(config_path) / "config.json").is_file():
        raise ValueError("Flash-Next crop requires a local directory containing the released config.json")
    config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen4_exp":
        raise ValueError("Expected the Qwen3.8-Flash-Next qwen4_exp configuration")
    text = config.text_config
    text.hidden_size = text.ple_embed_dim = config.vision_config.out_hidden_size = 64
    text.moe_intermediate_size = text.shared_expert_intermediate_size = 32
    text.hc_lowrank = 8
    text.num_experts = 16
    if text.num_experts_per_tok > text.num_experts:
        raise ValueError("Released routing top-k exceeds the selected crop expert count")
    text.num_attention_heads = 4
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 6
    text.linear_key_head_dim = text.linear_value_head_dim = 16
    text.head_dim = 64
    text.indexer_head_dim = 32
    text.rope_parameters["mrope_section"] = [3, 3, 2]
    text.ngram_vocab_size_base = 127
    text.vocab_size = 512
    text.eos_token_id = text.bos_token_id = 511
    text.use_cache = False
    if sparse_budget is not None:
        if sparse_budget < text.indexer_compress_ratio or sparse_budget % text.indexer_compress_ratio:
            raise ValueError("Crop indexer budget must be a positive multiple of the compression ratio")
        text.indexer_budget = sparse_budget
    config.image_token_id = 500
    config.video_token_id = 501
    config.vision_start_token_id = 502
    config.vision_end_token_id = 503
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 128
    config.vision_config.num_heads = 4
    return config


def build_crop(config_path: str, *, sparse_budget: int | None = None, **kwargs: Any) -> Any:
    """Build the crop through the normal HyperParallel construction and sharding pipeline."""
    config = cropped_config(config_path, sparse_budget=sparse_budget)
    config._attn_implementation = "sdpa"  # pylint: disable=protected-access
    config._experts_implementation = "eager"  # pylint: disable=protected-access
    return HyperAutoModelForImageTextToText.from_config(config, attn_implementation="sdpa", **kwargs)

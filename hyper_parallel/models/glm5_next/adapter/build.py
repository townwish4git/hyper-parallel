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
"""Construct the GLM training architecture through HyperParallel's model builder."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from transformers import Glm5NextConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping

from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models import HyperAutoModelForImageTextToText
from hyper_parallel.models.glm5_next.adapter.context_parallel import Glm5NextCPTrainingModel
from hyper_parallel.models.glm5_next.adapter.replacements import Glm5NextAlignedSparseAttention
from hyper_parallel.models.glm5_next.modeling_glm5_next import Glm5NextTrainingModel, build_text_attention


# Both training variants have the parameter layout that Transformers derives for GLM.
for _architecture in (Glm5NextTrainingModel, Glm5NextCPTrainingModel):
    register_checkpoint_conversion_mapping(
        _architecture.__name__, get_checkpoint_conversion_mapping("glm5_next"), overwrite=True,
    )


def _construct_training_model(
    config: Glm5NextConfig, *, mesh: MeshContext | None = None,
) -> Glm5NextTrainingModel:
    if mesh is not None and mesh.cp_size > 1:
        return Glm5NextCPTrainingModel(config, cp=CPContext(mesh.cp_mesh))
    return Glm5NextTrainingModel(
        config, attention_factory=partial(build_text_attention, sparse=Glm5NextAlignedSparseAttention),
    )


def load_config(
    config_path: str, *, indexer_mode: str = "frozen", mtp_weight: float = 0.1,
    kpool_weight: float = 1.0, ce_chunk_size: int = 256,
) -> Glm5NextConfig:
    """Read local config and configure model-owned training options."""
    path = Path(config_path).expanduser()
    path = path / "config.json" if path.is_dir() else path
    config = Glm5NextConfig.from_json_file(str(path))
    config.architectures = ["Glm5NextForConditionalGeneration"]
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    config._experts_implementation = "eager"  # pylint: disable=protected-access
    config.text_config.use_cache = False
    config.hp_indexer_mode = indexer_mode
    config.hp_mtp_weight = mtp_weight
    config.hp_kpool_weight = kpool_weight
    config.hp_ce_chunk_size = ce_chunk_size
    return config


def build_glm5_next(
    config_path: str | None = None, *, checkpoint_path: str | None = None, indexer_mode: str = "frozen",
    mtp_weight: float = 0.1, kpool_weight: float = 1.0, ce_chunk_size: int = 256, trust_remote_code: bool = False,
    **kwargs: Any,
) -> Glm5NextTrainingModel:
    """Create a scratch or pretrained GLM using the ordinary VLM Trainer path."""
    if trust_remote_code:
        raise ValueError("GLM training uses the local Transformers implementation")
    if (config_path is None) == (checkpoint_path is None):
        raise ValueError("Supply either a local GLM scratch config or a local checkpoint directory")
    setup = kwargs.get("distributed_setup")
    if setup is not None:
        mesh = setup.mesh_context
        if mesh.tp_size != 1 or mesh.pp_size != 1:
            raise ValueError("GLM training currently supports CP, EP and FSDP at TP1/PP1")
    if checkpoint_path is not None:
        directory = Path(checkpoint_path).expanduser().resolve()
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"GLM checkpoint config does not exist: {directory}")
        return HyperAutoModelForImageTextToText.from_pretrained(
            str(directory), model_factory=_construct_training_model, attn_implementation="eager",
            local_files_only=True, config_overrides={
                "hp_indexer_mode": indexer_mode, "hp_mtp_weight": mtp_weight,
                "hp_kpool_weight": kpool_weight, "hp_ce_chunk_size": ce_chunk_size,
            }, **kwargs,
        )
    config = load_config(config_path, indexer_mode=indexer_mode, mtp_weight=mtp_weight,
                         kpool_weight=kpool_weight, ce_chunk_size=ce_chunk_size)
    return HyperAutoModelForImageTextToText.from_config(
        config, model_factory=_construct_training_model, attn_implementation="eager", **kwargs,
    )

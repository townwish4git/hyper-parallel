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
"""Full model specifications and deterministic inputs for optimizer-only measurements."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from hyper_parallel import fully_shard
from hyper_parallel.components.optim.parameter_groups import split_muon_adamw_params
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement import (
    replace_qwen3_moe_flash_attention,
    replace_qwen3_moe_grouped_experts,
)


MODEL_NUMEL = {"qwen3-30b-a3b": 30532122624, "qwen3.5-27b": 26895998464}


def model_config_path(name: str) -> Path:
    """Return the frozen official model specification used by this benchmark.

    Args:
        name: Supported model specification name.

    Returns:
        Path of its frozen configuration.
    """
    return Path(__file__).with_name("model_configs") / (name + ".json")


def _model_on_meta(name, layers):
    payload = json.loads(model_config_path(name).read_text())
    if name == "qwen3.5-27b":
        config = Qwen3_5TextConfig(**payload["text_config"])
        if layers:
            config.num_hidden_layers = layers
            config.layer_types = config.layer_types[:layers]
        config._attn_implementation = "eager"
        return Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16)
    config = Qwen3MoeConfig(**payload)
    if layers:
        config.num_hidden_layers = layers
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16)
    for index, layer in enumerate(model.model.layers):
        fqn = f"model.layers.{index}"
        layer.self_attn = replace_qwen3_moe_flash_attention(
            module=layer.self_attn, module_fqn=f"{fqn}.self_attn", context={})
        layer.mlp.experts = replace_qwen3_moe_grouped_experts(
            module=layer.mlp.experts, module_fqn=f"{fqn}.mlp.experts", context={})
    return model


def build_model(name: str, mesh: object, layers: int = 0) -> tuple[torch.nn.Module, list, list, dict]:
    """Materialize sharded BF16 weights without pretrained weights or a forward pass.

    Args:
        name: Supported model specification name.
        mesh: Sharding device mesh.
        layers: Diagnostic layer limit; zero uses the full specification.

    Returns:
        Model, Muon parameters, parameter names and model metadata.
    """
    with torch.device("meta"):
        model = _model_on_meta(name, layers)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True, comm_fusion=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=True, comm_fusion=False)
    model.to_empty(device="npu")
    schedulers = {id(module.hsdp_scheduler): module.hsdp_scheduler for module in model.modules()
                  if hasattr(module, "hsdp_scheduler")}
    for scheduler in schedulers.values():
        scheduler.hsdp_state.lazy_init()
    matrices, remaining, names, _ = split_muon_adamw_params(model)
    description = {"full_numel": sum(p.numel() for p in model.parameters()),
                   "muon_numel": sum(p.numel() for p in matrices),
                   "excluded_adamw_numel": sum(p.numel() for p in remaining),
                   "parameters": [{"name": name, "shape": list(p.shape),
                                   "local_shape": list(to_local_if_dtensor(p).shape)}
                                  for name, p in zip(names, matrices)]}
    if not layers and description["full_numel"] != MODEL_NUMEL[name]:
        raise AssertionError(description)
    return model, matrices, names, description


@torch.no_grad()
def initialize_parameters(parameters: list, seed: int) -> None:
    """Reset identical parameter shards before each independent optimizer run.

    Args:
        parameters: Local or sharded optimizer parameters.
        seed: Shared initialization seed.
    """
    for index, parameter in enumerate(parameters):
        generator = torch.Generator(device="npu").manual_seed(seed + index * 1024 + dist.get_rank())
        to_local_if_dtensor(parameter).normal_(0, 0.02, generator=generator)


@torch.no_grad()
def initialize_gradients(parameters: list, seed: int, step: int = 0) -> None:
    """Supply reproducible synthetic gradients without touching the optimizer implementation.

    Args:
        parameters: Local or sharded optimizer parameters.
        seed: Shared initialization seed.
        step: Gradient sequence index.
    """
    for index, parameter in enumerate(parameters):
        generator = torch.Generator(device="npu").manual_seed(
            seed + 10_000_000 + index * 1024 + step * 1_000_000 + dist.get_rank())
        if parameter.grad is None:
            parameter.grad = torch.empty_like(parameter)
        to_local_if_dtensor(parameter.grad).normal_(0, 0.05, generator=generator)

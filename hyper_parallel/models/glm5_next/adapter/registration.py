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
"""Register model-owned integration providers."""

from typing import Any

from torch import nn

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


def _sharding(*, model: nn.Module, context: Any) -> dict[str, ModuleShardingSpec]:
    """Own the replicated GLM branches and leave routed experts to the EP planner."""
    language = "model.language_model"
    roots = ["lm_head", "model.visual", f"{language}.embed_tokens", f"{language}.norm", f"{language}.hc_head"]
    for index, layer in enumerate(model.get_submodule(f"{language}.layers")):
        prefix = f"{language}.layers.{index}"
        if not hasattr(layer.mlp, "experts"):
            roots.append(prefix)
            continue
        roots.extend(f"{prefix}.{name}" for name, _ in layer.named_children() if name != "mlp")
        roots.extend(f"{prefix}.mlp.{name}" for name, _ in layer.mlp.named_children() if name != "experts")

    replicated = {"tp": Replicate()} if "tp" in context.axis_sizes else {}
    layouts = {}
    for root in roots:
        params = {name: replicated.copy() for name, _ in model.get_submodule(root).named_parameters()}
        if params:
            layouts[root] = ModuleShardingSpec(params=params, in_src={}, in_dst={}, is_boundary=False)
    return layouts


def _load_validation():
    """Load validation-only declarations lazily."""
    from .validation import get_validation_spec  # pylint: disable=C0415

    return get_validation_spec()


def _fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Expose decoder, MTP and vision execution units to the common FSDP planner."""
    return (
        *(f"model.visual.blocks.{index}" for index in range(len(model.model.visual.blocks))),
        *(f"model.language_model.layers.{index}" for index in range(len(model.model.language_model.layers))),
    )


def _fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Exclude the independently declared vision tower from decoder discovery."""
    del model
    return ("model.visual",)


def _fsdp_order(model: Any, selected: tuple[str, ...]) -> tuple[str, ...]:
    """Prefetch vision before the language decoder whenever both are present."""
    order = _fsdp_wrap_modules(model)
    return tuple(name for name in order if name in selected) + tuple(name for name in selected if name not in order)


MODEL_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Glm5NextForConditionalGeneration",
    model_type="glm5_next",
    sharding=_sharding,
    validation=_load_validation,
    fsdp_wrap_modules=_fsdp_wrap_modules,
    fsdp_excluded_subtrees=_fsdp_excluded_subtrees,
    fsdp_execution_order=_fsdp_order,
)
register_model_adapter(MODEL_ADAPTER_SPEC)

register_custom_model(
    "Glm5NextForConditionalGeneration",
    "hyper_parallel.models.glm5_next.adapter.build",
    "Glm5NextTrainingModel",
)

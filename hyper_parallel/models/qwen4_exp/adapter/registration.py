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

from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


def _load_validation():
    """Load validation-only declarations lazily."""
    from .validation import get_validation_spec  # pylint: disable=C0415

    return get_validation_spec()


def _sharding_rules() -> list[tuple[list[str], ParamRole]]:
    """Keep scalar gates and GDN state parameters replicated on CP."""
    return [(["shared_expert_gate", ".linear_attn.A_log", ".linear_attn.dt_bias"], ParamRole.REPLICATED)]


def _register_rotary_buffers(model):
    """Load the Transformers-dependent materialization implementation on demand."""
    from .materialization import register_rotary_buffers  # pylint: disable=C0415

    register_rotary_buffers(model)


MODEL_ADAPTER_SPEC = ModelAdapterSpec(
    materialization=_register_rotary_buffers,
    sharding_rules=_sharding_rules,
    architecture='Qwen4ExpForConditionalGeneration',
    model_type='qwen4_exp',
    validation=_load_validation,
)
register_model_adapter(MODEL_ADAPTER_SPEC)
for architecture in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForTraining"):
    register_custom_model(
        architecture,
        "hyper_parallel.models.qwen4_exp.modeling_qwen4_exp",
        "Qwen4ExpForTraining",
    )

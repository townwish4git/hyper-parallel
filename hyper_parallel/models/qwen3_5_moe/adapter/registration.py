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
"""Register native Qwen3.5-MoE text and multimodal materialization adapters."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _register_materialization(model):
    """Load the optional HF implementation only when constructing a model."""
    from hyper_parallel.models.qwen3_5_moe.adapter.materialization import (  # pylint: disable=C0415
        register_materialization,
    )

    register_materialization(model)


QWEN3_5_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5MoeForConditionalGeneration",
    model_type="qwen3_5_moe",
    materialization=_register_materialization,
)

QWEN3_5_MOE_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5MoeForCausalLM",
    model_type="qwen3_5_moe_text",
    materialization=_register_materialization,
)

register_model_adapter(QWEN3_5_MOE_ADAPTER_SPEC)
register_model_adapter(QWEN3_5_MOE_TEXT_ADAPTER_SPEC)

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
"""Contiguous training CP for Flash-Next's sequence-dependent decoder branches."""

from collections.abc import Callable
from functools import wraps
from typing import Any

from torch import Tensor, nn

from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.recipe_spec import inner_wrapper


def _local_slice(value: Tensor, context: CPContext) -> Tensor:
    if value.shape[1] % context.size:
        raise ValueError("Flash-Next requires sequence length divisible by CP size")
    partition = context.partition(value.shape[1] // context.size)
    return value.narrow(1, partition.offset, partition.lengths[partition.rank]).contiguous()


@inner_wrapper
def shard_decoder_input(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Partition the backbone activation after HF's multimodal fusion and RoPE setup."""
    del mesh, tp_mesh, ep_mesh
    context = CPContext(cp_mesh)
    original = target_module.forward

    @wraps(original)
    def compute(hidden_states, *args, **kwargs):
        return original(_local_slice(hidden_states, context), *args, **kwargs)

    return compute


@inner_wrapper
def gather_mixer_output(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Restore the global token order for the shared LM head and causal objective."""
    del mesh, tp_mesh, ep_mesh
    context = CPContext(cp_mesh)
    original = target_module.forward

    @wraps(original)
    def compute(hyper_input):
        return context.gather_sequence(original(hyper_input))[0]

    return compute


@inner_wrapper
def gather_sequence_compute(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Reconstruct the causal history for GDN, QSA or PLE before invoking native math."""
    del mesh, tp_mesh, ep_mesh
    context = CPContext(cp_mesh)
    original = target_module.forward

    @wraps(original)
    def compute(hidden_states, *args, **kwargs):
        if kwargs.get("cache") is not None:
            raise ValueError("Flash-Next context parallelism requires use_cache=false")
        full, part = context.gather_sequence(hidden_states)
        output = original(full, *args, **kwargs)
        length = hidden_states.shape[1]
        if isinstance(output, tuple):
            states, weights = output
            local_weights = None if weights is None else weights.narrow(-2, part.offset, length)
            return states.narrow(1, part.offset, length).contiguous(), local_weights
        return output.narrow(1, part.offset, length).contiguous()

    return compute


@inner_wrapper
def shard_mtp_embeddings(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Pair each rank's MTP activations with its globally shifted embeddings."""
    del mesh, tp_mesh, ep_mesh
    context = CPContext(cp_mesh)
    original = target_module.forward

    @wraps(original)
    def compute(hidden_states, next_embeddings, position_ids, attention_mask):
        return original(hidden_states, _local_slice(next_embeddings, context), position_ids, attention_mask)

    return compute

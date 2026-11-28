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
"""Qwen3.8-Flash-Next training model with the released one-step MTP head."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from transformers import initialization as hf_init
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.core_model_loading import Concatenate, WeightConverter
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpPreTrainedModel,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextModel,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextTopKRouter,
)


from hyper_parallel.components.modules import GatedDeltaNetModule
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor


class _Qwen4ExpTrainingInitialization:
    """Initialize direct router weights and support the GDN FSDP layout."""

    config: Qwen4ExpTextConfig

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, Qwen4ExpTextTopKRouter):
            # HF marks child parameters initialized before visiting their enclosing MoE block.
            hf_init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (Qwen4ExpTextGatedDeltaNet, GatedDeltaNetModule)):
            hf_init.ones_(module.dt_bias)
            a_log = module.A_log
            values = torch.empty(module.num_v_heads, device=a_log.device, dtype=a_log.dtype).uniform_(0.01, 16).log_()
            if isinstance(a_log, DTensor):
                values = distribute_tensor(values, a_log.device_mesh, a_log.placements)
            hf_init.copy_(a_log, values)
        else:
            # The concrete HF text model supplies the native initializer.
            super()._init_weights(module)  # pylint: disable=no-member


class _Qwen4ExpTrainingTextModel(_Qwen4ExpTrainingInitialization, Qwen4ExpTextModel):
    """Use HF text computation with Flash-Next's materialized initialization."""


class _Qwen4ExpTrainingBackbone(Qwen4ExpModel):
    """Keep native multimodal forward with a shard-aware text initializer."""

    def __init__(self, config: Qwen4ExpConfig) -> None:
        # Construct each weight once; the upstream constructor allocates a different child.
        Qwen4ExpPreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = _Qwen4ExpTrainingTextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen4ExpMultiTokenPrediction(_Qwen4ExpTrainingInitialization, Qwen4ExpPreTrainedModel):
    """Predict one extra token from the decoder's pre-mixer streams and next-token embedding."""

    def __init__(self, config: Qwen4ExpTextConfig) -> None:
        """Construct the draft layer and its independently parameterized residual mixer."""
        draft = deepcopy(config)
        draft.num_hidden_layers = 1
        draft.layer_types = ["indexed_attention"]
        draft.ple_layer_ids = []
        draft.rope_parameters["rope_theta"] = config.mtp["rope_theta"]
        draft._experts_implementation = "eager"  # pylint: disable=protected-access
        super().__init__(draft)
        width = config.hidden_size
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(config.hc_count * width, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(width, eps=config.rms_norm_eps)
        self.fc_embedding = nn.Linear(width, width, bias=False)
        self.fc_hidden = nn.Linear(width, width, bias=False)
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(draft, 0)])
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(draft, use_combine=False)
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(draft)
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply MTP fusion, QSA and MoE to teacher-forced input embeddings."""
        streams = self.pre_fc_norm_hidden(hidden_states).unflatten(-1, (-1, self.config.hidden_size))
        next_embeddings = self.fc_embedding(self.pre_fc_norm_embedding(next_embeddings))
        hidden_states = (self.fc_hidden(streams) + next_embeddings.unsqueeze(-2)).flatten(-2)
        rope = self.rotary_emb(hidden_states, position_ids)
        hidden_states = self.layers[0](hidden_states, position_embeddings=rope, attention_mask=attention_mask)
        return self.hyper_connection_mixer(hidden_states)


class Qwen4ExpForTraining(Qwen4ExpForConditionalGeneration):
    """Use HF for the multimodal backbone and add the published MTP topology."""

    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, config: Qwen4ExpConfig) -> None:
        """Extend the native conditional-generation model with the MTP branch."""
        # Construct each weight once; the upstream constructor allocates a different child.
        Qwen4ExpPreTrainedModel.__init__(self, config)  # pylint: disable=non-parent-init-called
        text = config.text_config
        self.model = _Qwen4ExpTrainingBackbone(config)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        text.tie_last_hidden_states = False
        if text.mtp_num_hidden_layers != 1 or text.mtp_use_dedicated_embeddings:
            raise ValueError("Flash-Next training requires one MTP layer and shared embeddings")
        self.mtp = Qwen4ExpMultiTokenPrediction(text)
        self.post_init()

    @classmethod
    def from_config(cls, config: Qwen4ExpConfig, **kwargs: Any) -> "Qwen4ExpForTraining":
        """Construct the training topology through the HyperAutoModel custom-model interface."""
        return cls._from_config(config, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Qwen4ExpCausalLMOutputWithPast | tuple:
        """Compute the standard causal objective and the optional teacher-forced draft loss."""
        mtp_weight = getattr(self.config, "hp_mtp_loss_weight", 0.1)
        if labels is None or mtp_weight == 0:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, **kwargs,
            )
        if input_ids is None or input_ids.shape[1] < 3:
            raise ValueError("MTP supervision requires input_ids with at least three positions")
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        if (kwargs.get("cache") is not None or kwargs.get("use_cache")
                or isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0):
            raise ValueError("MTP supervision requires uncached full-sequence training")
        return_dict = kwargs.pop("return_dict", self.config.return_dict)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        kwargs.pop("use_cache", None)
        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids, None, image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=kwargs.get("video_grid_thw"), attention_mask=attention_mask,
                mm_token_type_ids=kwargs.get("mm_token_type_ids"),
            )
        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            labels=labels, output_hidden_states=True if output_hidden_states else (-1,),
            use_cache=False, return_dict=True, **kwargs,
        )
        draft_logits = self._draft_logits(input_ids, outputs.hidden_states[-1], attention_mask, position_ids)
        targets = labels[:, 2:].reshape(-1)
        draft_sum = F.cross_entropy(
            draft_logits[:, :-2].reshape(-1, draft_logits.shape[-1]).float(),
            targets, ignore_index=-100, reduction="sum",
        )
        outputs.loss = outputs.loss + mtp_weight * draft_sum / targets.ne(-100).sum().clamp_min(1)
        if not output_hidden_states:
            outputs.hidden_states = None
        return outputs if return_dict else outputs.to_tuple()

    def _draft_logits(self, input_ids, hidden_states, attention_mask, position_ids):
        next_ids = F.pad(input_ids[:, 1:], (0, 1), value=0)
        next_embeddings = self.get_input_embeddings()(next_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        text_positions = position_ids[0] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else None
        if position_ids.ndim == 2:
            text_positions = position_ids
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif text_positions is not None:
            position_ids = position_ids[1:]
        mask = create_causal_mask(
            self.mtp.config, next_embeddings, attention_mask, cache=None,
            position_ids=text_positions, allow_is_causal_skip=False,
        )
        return self.lm_head(self.mtp(hidden_states, next_embeddings, position_ids, mask))


# The official experts and MTP tensors are already packed. Only PLE rows are split.
register_checkpoint_conversion_mapping(
    Qwen4ExpForTraining.__name__,
    [WeightConverter(
        source_patterns="ngram_embedding.shard_*.weight",
        target_patterns="ngram_embedding.weight",
        operations=[Concatenate(dim=0, num_shards_attribute="split_ngram_parts")],
        force_cpu=True,
    )],
)

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
"""Deterministic, tensor-only Omni replay for model integration and DCP continuation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from hyper_parallel.data.omni.omni_transform import OmniDataTransform


class Glm5NextReplayDataset(Dataset):
    """Read numbered, already packed and processor-encoded records from safetensors."""

    def __init__(self, data_path: str) -> None:
        """Index the archive without loading its tensor payload."""
        self.path = Path(data_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"GLM replay file does not exist: {self.path}")
        fields: dict[int, list[str]] = defaultdict(list)
        with safe_open(str(self.path), framework="pt", device="cpu") as archive:
            for key in archive.keys():
                prefix, separator, field = key.partition("/")
                if not separator or not prefix.isdecimal() or not field:
                    raise ValueError(f"Invalid GLM replay key: {key!r}")
                fields[int(prefix)].append(field)
        if sorted(fields) != list(range(len(fields))) or not fields:
            raise ValueError("GLM replay sample IDs must be contiguous integers starting at zero")
        self.fields = tuple(tuple(fields[index]) for index in range(len(fields)))

    def __len__(self) -> int:
        """Return the number of packed records."""
        return len(self.fields)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Read one record without decoding or downloading media."""
        names = self.fields[index]
        if index < 0:
            index += len(self)
        with safe_open(str(self.path), framework="pt", device="cpu") as archive:
            return {name: archive.get_tensor(f"{index}/{name}") for name in names}


def build_replay_dataset(data_path: str, **kwargs: Any) -> Glm5NextReplayDataset:
    """Expose offline replay through the normal VLM Trainer target contract."""
    del kwargs
    return Glm5NextReplayDataset(data_path)


def build_replay_assets(
    pretrained_model_name_or_path: str, trust_remote_code: bool = False,
) -> SimpleNamespace:
    """Attach tokenizer metadata to an already encoded replay dataset."""
    if trust_remote_code:
        raise ValueError("GLM offline replay requires the local tokenizer implementation")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, local_files_only=True, trust_remote_code=False,
    )
    return SimpleNamespace(tokenizer=tokenizer, chat_template=tokenizer.chat_template)


class Glm5NextReplayTransform(OmniDataTransform):
    """Declare the length budget for online transport of already encoded batches."""

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return the record exactly as stored by the offline processor."""
        return sample

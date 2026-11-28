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
"""Deterministic multimodal overfit fixture for the dimension-cropped model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from hyper_parallel.data.omni.omni_transform import OmniDataTransform


@dataclass(frozen=True)
class CropProcessor:
    """Local tokenizer and image processor used by the standard VLM Trainer."""

    tokenizer: Any
    image_processor: Any


def build_crop_processor(config_path: str) -> CropProcessor:
    """Load the released tokenizer and preprocessing geometry from local assets."""
    source = Path(config_path)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    image_processor = AutoImageProcessor.from_pretrained(
        source, backend="pil", local_files_only=True, trust_remote_code=False,
    )
    return CropProcessor(tokenizer=tokenizer, image_processor=image_processor)


class CropTransform(OmniDataTransform):
    """Encode one real image and emit in-vocabulary tokens for the crop."""

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Use a single patch-merged image token followed by a repeated text target."""
        with Image.open(sample["image_path"]) as source:
            image = source.convert("RGB").resize((32, 32))
        media = self.processor.image_processor(images=image, do_resize=False, return_tensors="pt")
        ids = torch.tensor([11, 502, 500, 503, 12, 13, 14, 23, 24, 25, 23, 24, 25, 23, 24, 511])
        labels = ids.clone()
        labels[:7] = -100
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": ids.eq(500).int(),
            "pixel_values": media["pixel_values"],
            "image_grid_thw": media["image_grid_thw"],
        }


class RepeatedImageCaption(Dataset):
    """Expose a deterministic overfit set to the standard fixed-batch loader."""

    def __init__(self, image_path: str, transform: CropTransform, samples: int = 256) -> None:
        """Encode the fixed image sample once and keep it in memory."""
        if not Path(image_path).is_file():
            raise ValueError(f"Flash-Next validation image does not exist: {image_path}")
        self.sample = transform.encode_sample({"image_path": image_path})
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of repeated records in the epoch."""
        return self.samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the encoded example for every valid index."""
        if not 0 <= index < self.samples:
            raise IndexError(index)
        return self.sample

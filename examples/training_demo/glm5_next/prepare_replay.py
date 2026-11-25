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
"""Encode local text/image/video chat records for deterministic GLM Trainer validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from hyper_parallel.data.constants import OMNI_CONCAT_FIELDS, OMNI_CP_TOKEN_FIELDS, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.omni.omni_transform import AutoProcessorTransform, build_auto_processor


def pack_documents(documents: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate complete documents, preserving raw boundaries and media order."""
    lengths = torch.tensor([doc["input_ids"].numel() for doc in documents], dtype=torch.int32)
    packed = {"cu_seq_lens": torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))}
    for name in OMNI_CP_TOKEN_FIELDS | OMNI_CONCAT_FIELDS:
        values = [doc[name] for doc in documents if name in doc]
        if not values:
            continue
        if name in OMNI_CP_TOKEN_FIELDS and len(values) != len(documents):
            if name != "mm_token_type_ids":
                raise ValueError(f"Token field {name!r} is missing in one or more documents")
            values = [doc.get(name, torch.zeros_like(doc["input_ids"])) for doc in documents]
        packed[name] = torch.cat(values)
    return packed


def main() -> None:
    """Read JSONL messages and publish one tensor-only replay archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--documents-per-record", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if min(args.documents_per_record, args.repeat) < 1:
        parser.error("documents-per-record and repeat must be positive")
    processor = build_auto_processor(
        pretrained_model_name_or_path=args.processor, local_files_only=True, trust_remote_code=False,
    )
    transform = AutoProcessorTransform(processor=processor, max_seq_len=args.max_seq_len)
    with args.input.open(encoding="utf-8") as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    if not samples:
        parser.error("input must contain at least one JSONL training record")
    for sample in samples:
        sample[ONLINE_SOURCE_PATH_KEY] = str(args.input.resolve())
    encoded = [transform.encode_sample(sample) for sample in samples]
    records = [
        pack_documents(encoded[begin:begin + args.documents_per_record])
        for begin in range(0, len(encoded), args.documents_per_record)
    ]
    if any(record["input_ids"].numel() > args.max_seq_len for record in records):
        parser.error("packed record exceeds max-seq-len")
    fields = {
        f"{index}/{name}": value.contiguous().clone()
        for index, record in enumerate(records * args.repeat)
        for name, value in record.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(fields, str(args.output))


if __name__ == "__main__":
    main()

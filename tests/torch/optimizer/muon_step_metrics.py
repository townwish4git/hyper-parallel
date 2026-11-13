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
"""Chunked all-element precision statistics; no hooks into optimizer arithmetic."""

import math

import torch


_FIELDS = ("elements", "different", "different_bit_patterns", "nonfinite",
           "actual_squared", "reference_squared", "error_squared", "dot")


def empty_metrics() -> dict:
    """Create additive global statistics with a separate maximum error."""
    return dict.fromkeys((*_FIELDS, "max_abs"), 0)


def merge_metrics(target: dict, source: dict) -> None:
    """Merge disjoint parameter shards without averaging their relative errors.

    Args:
        target: Accumulator updated in place.
        source: Statistics for disjoint elements.
    """
    for key in _FIELDS:
        target[key] += source[key]
    target["max_abs"] = max(target["max_abs"], source["max_abs"])


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor, initial: torch.Tensor = None) -> dict:
    """Measure every CPU element, optionally after subtracting a common initial value.

    Args:
        actual: Candidate CPU tensor.
        reference: Baseline CPU tensor.
        initial: Optional common initial tensor for effective updates.

    Returns:
        Additive statistics over every element.
    """
    result = empty_metrics()
    if actual.shape != reference.shape:
        raise ValueError("Precision tensors have different shapes")
    actual, reference = actual.reshape(-1), reference.reshape(-1)
    initial = None if initial is None else initial.reshape(-1)
    for start in range(0, actual.numel(), 1_048_576):
        left = actual[start:start + 1_048_576].float()
        right = reference[start:start + 1_048_576].float()
        if initial is not None:
            origin = initial[start:start + 1_048_576].float()
            left, right = left - origin, right - origin
        delta = left - right
        raw_left = actual[start:start + 1_048_576] if initial is None else left
        raw_right = reference[start:start + 1_048_576] if initial is None else right
        word = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[raw_left.element_size()]
        bits = raw_left.view(word) != raw_right.view(word)
        partial = {"different_bit_patterns": int(torch.count_nonzero(bits)),
                   "elements": left.numel(), "different": int(torch.count_nonzero(left != right)),
                   "nonfinite": int(torch.count_nonzero(~(torch.isfinite(left) & torch.isfinite(right)))),
                   "actual_squared": float((left * left).sum(dtype=torch.float64)),
                   "reference_squared": float((right * right).sum(dtype=torch.float64)),
                   "error_squared": float((delta * delta).sum(dtype=torch.float64)),
                   "dot": float((left * right).sum(dtype=torch.float64)),
                   "max_abs": float(delta.abs().max())}
        merge_metrics(result, partial)
    return result


def summarize_metrics(stats: dict) -> dict:
    """Convert aggregate sums into relative L2 and cosine similarity.

    Args:
        stats: Aggregated element statistics.

    Returns:
        Relative L2, cosine, counts and maximum absolute difference.
    """
    denominator = math.sqrt(stats["reference_squared"])
    cosine_denominator = math.sqrt(stats["actual_squared"] * stats["reference_squared"])
    return {"elements": stats["elements"], "different": stats["different"],
            "different_bit_patterns": stats["different_bit_patterns"], "nonfinite": stats["nonfinite"],
            "max_abs": stats["max_abs"], "reference_l2": denominator,
            "relative_l2": math.sqrt(stats["error_squared"]) / max(denominator, 1e-30),
            "cosine": stats["dot"] / cosine_denominator if cosine_denominator else None}

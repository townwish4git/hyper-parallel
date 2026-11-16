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
"""Check fused symmetric products across tiling boundaries and fingerprint public Muon steps."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from hyper_parallel.components.functional.symmetric_matmul import symmetric_matmul
from hyper_parallel.core.optimizer.muon import Muon


def _digest(value):
    return hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def _products():
    reports = []
    for batch, n, k in ((1, 256, 256), (3, 512, 513), (2, 768, 160), (1, 257, 129)):
        for dtype in (torch.bfloat16, torch.float16):
            value = (torch.randn(batch, n, k) / k**0.5).to(dtype)
            additive = torch.randn(batch, n, n).to(dtype)
            additive = (additive + additive.mT).contiguous()
            reference = (2.927 * (value.double() @ value.double().mT) - 6.8946 * additive.double()).to(dtype)
            left, bias = value.npu(), additive.npu()
            backing = torch.full((batch * n * n + 2,), 123, device="npu", dtype=dtype)
            output = backing[1:-1].view(batch, n, n)
            symmetric_matmul(left, left.mT, bias, alpha=2.927, beta=-6.8946, out=output, backend="ascendc")
            actual = output.cpu()
            error = ((actual.float() - reference.float()).norm() / reference.float().norm()).item()
            if error >= 0.004 or not torch.equal(actual, actual.mT):
                raise AssertionError((batch, n, k, dtype, error))
            if backing[0].item() != 123 or backing[-1].item() != 123:
                raise AssertionError("Unaligned output overwrote guards")
            reports.append({"shape": [batch, n, k], "dtype": str(dtype), "relative_l2": error,
                            "sha256": _digest(output)})
    return reports


@torch.no_grad()
def _step(shape, mode):
    torch.manual_seed(51430)
    parameter = torch.nn.Parameter(torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.02)
    parameter.grad = torch.randn_like(parameter) * 0.05
    optimizer = Muon([parameter], lr=0.01, weight_decay=0.1, momentum=0.95, nesterov=True,
                     ns_steps=5, ns_variant="asym5", use_symm_gemm=mode != "cann",
                     use_symm_gemm_epilogue=mode == "epilogue")
    hashes = []
    for step in range(5):
        optimizer.step()
        if step in (0, 4):
            hashes.append({"step": step + 1, "parameter": _digest(parameter),
                           "momentum": _digest(optimizer.state[parameter]["momentum_buffer"])})
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    samples = []
    for _ in range(15):
        start = time.perf_counter()
        optimizer.step()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    result = {"shape": shape, "mode": mode, "median_ms": statistics.median(samples), "samples_ms": samples,
              "peak_allocated_bytes": torch.npu.max_memory_allocated(), "fingerprints": hashes}
    del optimizer, parameter
    torch.npu.empty_cache()
    return result


def main() -> None:
    """Validate boundary cases and measure complete local steps on representative Qwen tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CLOSE_MATMUL_K_SHIFT") != "1":
        raise ValueError("Set CLOSE_MATMUL_K_SHIFT=1 before process startup")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(43)
    report = {"scope": "Single parameter Muon.step, including momentum and update, without communication; "
                       "not a full-model end-to-end benchmark.", "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "products": _products(), "steps": []}
    for shape in ((32, 768, 2048), (5120, 17408)):
        for mode in ("cann", "symm", "epilogue"):
            result = _step(shape, mode)
            report["steps"].append(result)
            print(json.dumps(result), flush=True)
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

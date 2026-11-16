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
"""Measure and profile standard NS with the production symmetric/epilogue flags."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from hyper_parallel.core.optimizer.muon import zeropower_via_newtonschulz5


_MODES = {"cann": {}, "symm": {"use_symm_gemm": True},
          "epilogue": {"use_symm_gemm": True, "use_symm_gemm_epilogue": True}}
_SHAPES = {"expert": (32, 768, 2048), "dense": (1, 5120, 17408)}


def _measure(operation):
    for _ in range(3):
        operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def _profile(operation, path):
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
        record_shapes=True,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        ),
    ):
        for _ in range(2):
            operation()
        torch.npu.synchronize()


@torch.no_grad()
def main() -> None:
    """Collect unprofiled wall time and separate kernel traces for two representative NS shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CLOSE_MATMUL_K_SHIFT") != "1":
        raise ValueError("Set CLOSE_MATMUL_K_SHIFT=1 before starting the profiler")
    torch_npu.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(43)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    report = {"device": torch.npu.get_device_name(), "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "profiled_ns_calls_per_mode": 2, "cases": []}
    for name, shape in _SHAPES.items():
        value = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.01
        for mode, options in _MODES.items():
            operation = partial(zeropower_via_newtonschulz5, value, 5, **options)
            timing = _measure(operation)
            row = {"case": name, "shape": shape, "mode": mode, **timing}
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            _profile(operation, folder / name / mode)
        del value
    (folder / "timings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

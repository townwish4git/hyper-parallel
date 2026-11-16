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
"""Controlled native-ABI epilogue ablations with preallocated, identical inputs."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

import hyper_parallel


_MODES = {"direct_bf16": (1, 1.0, 0.0), "scratch_identity": (0, 1.0, 0.0),
          "scratch_scale": (0, 2.9270, 0.0), "scratch_polynomial": (0, 2.9270, -6.8946)}
_SHAPES = {"expert": (32, 768), "dense": (1, 5120)}


class _NativePlan(ctypes.Structure):
    _fields_ = [("tiling", ctypes.c_uint32 * 128)] + [
        (name, ctypes.c_int) for name in ("n", "k", "batch", "tile", "cores", "dtype", "direct", "row_tile")]


def _library(folder):
    for name in ("libhp_epilogue_kernel.so", "libhp_fused_kernel.so"):
        ctypes.CDLL(str(folder / name), mode=ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(folder / "libhp_symmetric.so"))
    library.hp_abi_version.restype = ctypes.c_int
    if library.hp_abi_version() != 4:
        raise RuntimeError("This diagnostic requires native ABI 4")
    library.hp_plan.argtypes, library.hp_plan.restype = [ctypes.c_int] * 7, ctypes.c_void_p
    library.hp_plan_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.hp_plan_info.restype = None
    library.hp_destroy.argtypes, library.hp_destroy.restype = [ctypes.c_void_p], None
    library.hp_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_float] * 2
    library.hp_launch.restype = ctypes.c_uint32
    if hasattr(library, "hp_onchip"):
        library.hp_onchip.argtypes, library.hp_onchip.restype = [ctypes.c_void_p], ctypes.c_int
    return library


class _Operation:
    def __init__(self, library, operand, transposed, output, scratch, mode):
        self.library = library
        self.direct, self.alpha, self.beta = _MODES[mode]
        batch, n, _ = operand.shape
        cores = torch.npu.get_device_properties(0).cube_core_num
        self.pointer = library.hp_plan(n, n, batch, 0, cores, 1, self.direct)
        if not self.pointer:
            raise RuntimeError("Native plan creation failed")
        values = (ctypes.c_int * 7)()
        library.hp_plan_info(self.pointer, values)
        raw = ctypes.cast(self.pointer, ctypes.POINTER(_NativePlan)).contents
        self.metadata = {"n": values[0], "k": values[1], "tile": values[2],
                         "base_shape": list(values[3:6]), "row_tile": values[6],
                         "cores": cores, "direct": self.direct, "tiling_words": list(raw.tiling)}
        self.metadata["onchip"] = bool(library.hp_onchip(self.pointer)) if hasattr(library, "hp_onchip") else False
        stream = torch_npu.npu.current_stream().npu_stream
        workspace = output if self.direct else scratch
        self.arguments = (self.pointer, stream, operand.data_ptr(), transposed.data_ptr(), operand.data_ptr(),
                          workspace.data_ptr(), output.data_ptr(), self.alpha, self.beta)

    def __call__(self):
        status = self.library.hp_launch(*self.arguments)
        if status:
            raise RuntimeError(f"Native launch failed: {status}")

    def close(self) -> None:
        """Release a plan after all queued kernels finish."""
        self.library.hp_destroy(self.pointer)


def _measure(operation):
    for _ in range(5):
        operation()
    torch.npu.synchronize()
    samples, wall_samples = [], []
    for _ in range(7):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(20):
            operation()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1e6 / 20)
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return {"median_us": statistics.median(samples), "samples_us": samples,
            "wall_median_us": statistics.median(wall_samples), "wall_samples_us": wall_samples}


def _profile(operation, folder):
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        ),
    ) as profiler:
        for _ in range(6):
            operation()
            torch.npu.synchronize()
            profiler.step()


def _case(library, batch, n, folder, profile):
    value = torch.randn(batch, n, n, device="npu", dtype=torch.bfloat16) / n**0.5
    value = (value + value.mT).contiguous()
    transposed = value.mT.contiguous()
    output = torch.empty_like(value)
    scratch = torch.empty(value.shape, device="npu", dtype=torch.float32)
    operations = {mode: _Operation(library, value, transposed, output, scratch, mode) for mode in _MODES}
    report = {"shape": list(value.shape), "plans": {mode: op.metadata for mode, op in operations.items()},
              "blocks": [], "identity_different_bits": None}
    for cycle in range(2):
        for mode in (list(_MODES) if cycle == 0 else list(reversed(_MODES))):
            row = {"mode": mode, "cycle": cycle, **_measure(operations[mode])}
            report["blocks"].append(row)
            print(json.dumps({"shape": report["shape"], **row}), flush=True)
    operations["direct_bf16"]()
    direct = output.clone()
    operations["scratch_identity"]()
    report["identity_different_bits"] = torch.count_nonzero(output.view(torch.int16) != direct.view(torch.int16)).item()
    operations["scratch_polynomial"]()
    report["polynomial_sha256"] = hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    report["polynomial_finite"] = bool(torch.isfinite(output).all())
    report["polynomial_symmetric"] = torch.equal(output, output.mT)
    if not report["polynomial_finite"] or not report["polynomial_symmetric"]:
        raise AssertionError(report)
    if profile:
        for mode, operation in operations.items():
            _profile(operation, folder / mode)
    torch.npu.synchronize()
    for operation in operations.values():
        operation.close()
    return report


@torch.no_grad()
def main() -> None:
    """Separate output/scratch costs from scaling and additive-input costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-library-dir", type=Path)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CLOSE_MATMUL_K_SHIFT") != "1":
        raise ValueError("Set CLOSE_MATMUL_K_SHIFT=1 before process startup")
    torch_npu.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(43)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    native = Path(hyper_parallel.__file__).parent / "components/functional/_ascendc/symmetric_matmul"
    native = args.native_library_dir or native
    library = _library(native)
    report = {"device": torch.npu.get_device_name(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "native_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in native.glob("*.so")},
              "cases": {}}
    for name, (batch, n) in _SHAPES.items():
        report["cases"][name] = _case(library, batch, n, folder / name, not args.no_profile)
        (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

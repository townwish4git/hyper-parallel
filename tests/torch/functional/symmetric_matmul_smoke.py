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
"""Installed-package Ascend C, automatic dispatch and Muon integration smoke tests."""

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import torch_npu

import hyper_parallel
from hyper_parallel.components.functional.symmetric_matmul import symmetric_matmul
from hyper_parallel.core.optimizer.muon import Muon, zeropower_via_newtonschulz5


_BRIDGE = importlib.import_module("hyper_parallel.components.functional.symmetric_matmul")


def _relative(actual, reference):
    return ((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-20)).item()


@torch.no_grad()
def _products():
    rows = []
    for shape in ((257, 513), (2, 1024, 4096), (32, 768, 768)):
        value = torch.randn(shape, device="npu", dtype=torch.bfloat16) / shape[-1]**0.5
        reference = symmetric_matmul(value, value.mT, backend="cann")
        native = symmetric_matmul(value, value.mT, backend="ascendc")
        automatic = symmetric_matmul(value, value.mT)
        expected = native if _BRIDGE._use_ascendc(value, 1) else reference
        assert torch.equal(automatic, expected), "Automatic dispatch did not select the expected backend"
        assert torch.equal(native, native.mT), "Native output is not bitwise symmetric"
        error = _relative(native, reference)
        assert error < 0.001, f"Unexpected pure GEMM discrepancy: {error}"
        rows.append({"shape": shape, "ascendc_selected": _BRIDGE._use_ascendc(value, 1), "relative_l2": error})
    value = torch.randn(257, 513, device="npu", dtype=torch.float16)
    backing = torch.full((257 * 257 + 2,), 123, device="npu", dtype=value.dtype)
    output = backing[1:-1].view(257, 257)
    stream = torch_npu.npu.Stream()
    stream.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(stream):
        symmetric_matmul(value, value.mT, out=output, backend="ascendc")
    torch_npu.npu.current_stream().wait_stream(stream)
    assert backing[0].item() == 123 and backing[-1].item() == 123
    assert torch.equal(output, output.mT)
    fused = symmetric_matmul(output, output, output, alpha=0.01, beta=-0.2, backend="ascendc")
    assert torch.isfinite(fused).all() and torch.equal(fused, fused.mT)
    return rows


@torch.no_grad()
def _optimizer():
    value = torch.randn(32, 768, 2048, device="npu", dtype=torch.bfloat16) * 0.01
    original = value.clone()
    update = zeropower_via_newtonschulz5(value, 5, use_symm_gemm=True)
    reference = zeropower_via_newtonschulz5(value, 5)
    different = torch.count_nonzero(update.view(torch.int16) != reference.view(torch.int16)).item()
    assert different == 0, f"NS different bit patterns={different}, expected=0"
    assert torch.equal(value, original), f"NS input changed for shape={tuple(value.shape)}"
    parameters = [torch.nn.Parameter(original.clone()) for _ in range(2)]
    optimizers = [Muon([parameter], use_symm_gemm=enabled)
                  for parameter, enabled in zip(parameters, (False, True))]
    for step in range(5):
        gradient = torch.randn_like(value)
        for parameter, optimizer in zip(parameters, optimizers):
            parameter.grad = gradient.clone()
            optimizer.step()
        different = torch.count_nonzero(
            parameters[0].view(torch.int16) != parameters[1].view(torch.int16)).item()
        assert different == 0, f"Step={step + 1}, different parameter bit patterns={different}, expected=0"
        left = optimizers[0].state[parameters[0]]["momentum_buffer"]
        right = optimizers[1].state[parameters[1]]["momentum_buffer"]
        assert torch.equal(left, right), f"Momentum differs at step={step + 1}"
    restored = Muon([parameters[1]])
    restored.load_state_dict(optimizers[1].state_dict())
    assert restored.param_groups[0]["use_symm_gemm"] is True, f"Restored groups={restored.param_groups}"
    return {"shape": list(value.shape), "different_bit_patterns": different,
            "optimizer_steps": 5, "step_and_state_restore": True}


def _require_precision_environment():
    if os.environ.get("CLOSE_MATMUL_K_SHIFT") != "1":
        raise RuntimeError("Set CLOSE_MATMUL_K_SHIFT=1 before starting the process")


def test_installed_symmetric_muon() -> None:
    """Worker entry for the repository single-card ST launcher."""
    _require_precision_environment()
    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    _products()
    _optimizer()
    torch_npu.npu.synchronize()


def main() -> None:
    """Verify the installed package location and exercise its bundled libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    location = Path(hyper_parallel.__file__).resolve()
    assert location.is_relative_to(Path(args.expected_root).resolve()), str(location)
    _require_precision_environment()
    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    report = {"package": str(location), "device": torch_npu.npu.get_device_name(0),
              "products": _products(), "muon": _optimizer()}
    torch_npu.npu.synchronize()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

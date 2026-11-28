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
"""On-device coverage for the Ascend C symmetric writeback paths."""

from unittest.mock import patch

import pytest
import torch
import torch_npu

from hyper_parallel.components.functional._ascendc.symmetric_matmul import (
    ascendc_fused_ns_step_out,
    ascendc_symmetric_gemm_out,
)
from hyper_parallel.core.optimizer import get_hyper_optimizer


@torch.no_grad()
def test_ascendc_symmetric_matmul() -> None:
    """Compare the queued product to native matmul and check off-diagonal mirror bits."""
    torch.manual_seed(42)
    torch_npu.npu.set_device(0)
    op = ascendc_symmetric_gemm_out()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for dtype in (torch.bfloat16, torch.float16):
            for batch, n, k in ((2, 513, 257), (32, 1024, 2048), (64, 512, 2048)):
                x = torch.randn(batch, k, n, dtype=dtype, device="npu") / k**0.5
                x = x.mT
                storage = x.new_full((batch * n * n + 2,), 17)
                out = storage[1:-1].view(batch, n, n)
                op(x, x.mT, out=out)
                expected = x @ x.mT
                torch.testing.assert_close(out, expected, rtol=0.008, atol=2e-5)
                torch.testing.assert_close(storage[[0, -1]], x.new_full((2,), 17))
                square = torch.empty_like(out)
                op(out, out, out=square)
                torch.testing.assert_close(square, out @ out, rtol=0.008, atol=2e-5)
                for result in (out, square):
                    bits = result.view(torch.int16)
                    for row in range(256, n, 256):
                        lower = bits[..., row:row + 256, :row]
                        upper = bits[..., :row, row:row + 256].mT
                        torch.testing.assert_close(lower, upper, rtol=0, atol=0)
        stream.synchronize()


@torch.no_grad()
def test_ascendc_fused_ns_step() -> None:
    """Compare queued fused steps with standalone SYMM GEMM and Torch writebacks."""
    torch.manual_seed(42)
    torch_npu.npu.set_device(0)
    device_name = torch.npu.get_device_name(0).lower()
    if not device_name.startswith(("ascend910b", "ascend910_93", "ascend950")):
        pytest.skip(f"Fused Newton-Schulz requires Ascend A2, A3, or A5, found {device_name}")
    symmetric_gemm = ascendc_symmetric_gemm_out()
    fused_ns_step = ascendc_fused_ns_step_out()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for shape in ((64, 128), (64, 1024), (2, 320, 448), (12, 576, 640)):
            x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
            x = x / x.norm(dim=(-2, -1), keepdim=True)
            expected_input = x.clone()
            out = torch.empty_like(x)
            gram = x.new_empty((*x.shape[:-1], x.size(-2)))
            poly = torch.empty_like(gram)
            ref_gram = torch.empty_like(gram)
            ref_square = torch.empty_like(gram)

            for a, b, c in ((1.5, -0.5, 0.25), (4.0848, -6.8946, 2.927)):
                symmetric_gemm(x, x.mT, out=ref_gram)
                symmetric_gemm(ref_gram, ref_gram, out=ref_square)
                ref_poly = (b * ref_gram) + (c * ref_square)
                expected = (a * x) + (ref_poly @ x)
                result = fused_ns_step(x, a, b, c, gram=gram, poly=poly, out=out)
                torch.testing.assert_close(gram, ref_gram, rtol=0, atol=0)
                torch.testing.assert_close(poly, ref_poly, rtol=0.008, atol=2e-5)
                torch.testing.assert_close(result, expected, rtol=0.016, atol=4e-5)
                torch.testing.assert_close(x, expected_input, rtol=0, atol=0)
                x, out = out, x
                expected_input.copy_(x)
    stream.synchronize()
    _check_muon_backend(fused_ns_step)
    if device_name.startswith("ascend950"):
        _check_fused_ns_multistream(fused_ns_step)


def _check_muon_backend(step_out) -> None:
    """Check real device dispatch through the public optimizer selector."""
    model = torch.nn.Linear(448, 320, bias=False, dtype=torch.bfloat16, device="npu")
    reference = torch.nn.Linear(448, 320, bias=False, dtype=torch.bfloat16, device="npu")
    reference.weight.copy_(model.weight)
    model.weight.grad = torch.rand_like(model.weight)
    reference.weight.grad = model.weight.grad.clone()
    native = get_hyper_optimizer(reference, [reference.weight], [])
    ascendc_optimizer = get_hyper_optimizer(model, [model.weight], [], muon_ns_backend="ascendc")
    native.step()
    with patch("hyper_parallel.core.optimizer.ascendc_ns.ascendc_fused_ns_step_out") as load:
        load.return_value.side_effect = step_out
        ascendc_optimizer.step()
        launches = load.return_value.call_count
        assert launches == 5, f"Expected five fused Newton–Schulz steps, got {launches}"
    torch.testing.assert_close(model.weight, reference.weight, rtol=0.016, atol=4e-5)


def _check_fused_ns_multistream(step_out) -> None:
    """Queue two full-core launches before waiting for either stream."""
    results = []
    for stream in (torch.npu.Stream(), torch.npu.Stream()):
        with torch.npu.stream(stream):
            x = torch.randn((12, 576, 640), dtype=torch.bfloat16, device="npu") / 1024
            gram = x.new_empty((12, 576, 576))
            out = step_out(x, 1., 0., 0., gram=gram, poly=torch.empty_like(gram), out=torch.empty_like(x))
            results.append((stream, x, out))
    for stream, x, out in results:
        stream.synchronize()
        torch.testing.assert_close(out, x, rtol=0, atol=0)

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

import torch
import torch_npu

from hyper_parallel.components.functional._ascendc.symmetric_matmul import ascendc_symmetric_gemm_out


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

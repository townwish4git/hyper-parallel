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

"""Run isolated Muon numerical comparisons and warmed device benchmarks.

Run as a module using an editable installation of this checkout. JSON output
contains failures as well as timings; performance does not imply numerical parity.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from hyper_parallel.core.optimizer.muon import zeropower_via_newtonschulz5
from hyper_parallel.core.optimizer.muon_pointwise import _apply_update_triton, _momentum_update_triton


def _errors(actual, expected):
    """Measure error without relative-error explosions near individual zeros."""
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    norm = expected.norm().clamp_min(1e-30)
    return {
        "relative_l2": (difference.norm() / norm).item(),
        "max_abs": difference.abs().max().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _timing(operation, repeats):
    """Report median and range over seven synchronized, warmed batches."""
    for _ in range(10):
        operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(repeats):
            operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / repeats)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)}


def _ns_cases(args):
    """Compare standard and fused matrix updates with fixed accuracy criteria."""
    rows = []
    shapes = ((256, 256), (256, 1024), (1024, 256), (4, 256, 1024), (512, 2048), (128, 8192))
    for dtype in (torch.float32, torch.bfloat16):
        for shape in shapes:
            torch.manual_seed(18)
            value = torch.randn(shape, device="npu", dtype=dtype)
            implementation = "gram" if args.scheme == "gram" else "standard_fused"
            for variant in ("legacy", "asym5"):
                def _baseline():
                    return zeropower_via_newtonschulz5(value, 5, ns_variant=variant)

                def _candidate():
                    return zeropower_via_newtonschulz5(value, 5, ns_variant=variant,
                                                    implementation=implementation,
                                                    gram_compute_dtype=args.gram_dtype)

                metrics = _errors(_candidate(), _baseline())
                limit = 1e-4 if dtype == torch.float32 else 0.05
                reference_time = _timing(_baseline, args.repeats)
                candidate_time = _timing(_candidate, args.repeats)
                row = {"scheme": args.scheme, "gram_dtype": args.gram_dtype,
                       "shape": shape, "dtype": str(dtype), "variant": variant,
                       "error": metrics, "limit": limit,
                       "passed": metrics["finite"] and metrics["relative_l2"] <= limit,
                       "baseline": reference_time, "candidate": candidate_time,
                       "speedup": reference_time["median_ms"] / candidate_time["median_ms"]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows



def _momentum_reference(grads, bufs, momentum1, momentum2, nesterov):
    updates = torch._foreach_add(grads, bufs, alpha=momentum1)
    torch._foreach_mul_(bufs, momentum2)
    torch._foreach_add_(bufs, grads)
    if nesterov:
        torch._foreach_mul_(updates, momentum1)
        torch._foreach_add_(updates, grads)
    return [update.to(torch.bfloat16) for update in updates]



def _merge_errors(metrics, actual, expected):
    for target, reference in zip(actual, expected):
        measured = _errors(target, reference)
        metrics["relative_l2"] = max(metrics["relative_l2"], measured["relative_l2"])
        metrics["max_abs"] = max(metrics["max_abs"], measured["max_abs"])
        metrics["finite"] &= measured["finite"]

def _momentum_cases(args):
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in ((256, 1024), (1024, 4096)):
            for nesterov in (False, True):
                torch.manual_seed(18)
                grads = [torch.randn(shape, dtype=dtype, device="npu") for _ in range(4)]
                old_grads = [grad.clone() for grad in grads]
                bufs = [torch.zeros_like(grad) for grad in grads]
                candidate_bufs = [buf.clone() for buf in bufs]
                error = {"relative_l2": 0.0, "max_abs": 0.0, "finite": True}
                state_error = dict(error)
                for _ in range(20):
                    expected = _momentum_reference(grads, bufs, .95, .85, nesterov)
                    actual = _momentum_update_triton(grads, candidate_bufs, .95, .85, nesterov)
                    _merge_errors(error, actual, expected)
                    _merge_errors(state_error, candidate_bufs, bufs)
                for grad, original in zip(grads, old_grads):
                    torch.testing.assert_close(grad, original, atol=0, rtol=0)
                assert all(out.data_ptr() != buf.data_ptr() for out, buf in zip(actual, candidate_bufs))
                baseline_time = _timing(lambda: _momentum_reference(grads, bufs, .95, .85, nesterov), args.repeats)
                candidate_time = _timing(
                    lambda: _momentum_update_triton(grads, candidate_bufs, .95, .85, nesterov), args.repeats)
                row = {"scheme": "momentum_triton", "shape": shape, "count": 4, "dtype": str(dtype),
                       "nesterov": nesterov, "momentum": [.95, .85], "error": error, "state_error": state_error,
                       "limit": .01, "passed": error["finite"] and state_error["finite"]
                       and error["relative_l2"] <= .01 and state_error["relative_l2"] <= .01,
                       "baseline": baseline_time, "candidate": candidate_time,
                       "speedup": baseline_time["median_ms"] / candidate_time["median_ms"]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def _pointwise_cases(args):
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((256, 1024), (1024, 4096)):
            torch.manual_seed(18)
            updates = [torch.randn(shape, dtype=dtype, device="npu") for _ in range(4)]
            params = [torch.randn(shape, dtype=torch.float32, device="npu") for _ in range(4)]
            actual = [param.clone() for param in params]
            if args.scheme == "writeback":
                params = [torch.empty_like(update) for update in updates]
                actual = [torch.empty_like(update) for update in updates]
                work = [update.clone() for update in updates]

                def _baseline():
                    for value, output in zip(work, params):
                        value.mul_(.99)
                        output.copy_(value)

                def _candidate():
                    for value, output in zip(updates, actual):
                        torch.mul(value, .99, out=output)
            else:
                def _baseline():
                    torch._foreach_mul_(params, .999)
                    torch._foreach_add_(params, updates, alpha=-.01)

                def _candidate():
                    _apply_update_triton(actual, updates, .999, -.01)
            _baseline()
            _candidate()
            metrics = {"relative_l2": 0.0, "max_abs": 0.0, "finite": True}
            _merge_errors(metrics, actual, params)
            baseline_time = _timing(_baseline, args.repeats)
            candidate_time = _timing(_candidate, args.repeats)
            row = {"scheme": args.scheme, "shape": shape, "count": 4, "dtype": str(dtype),
                   "error": metrics, "limit": 1e-6,
                   "passed": metrics["finite"] and metrics["relative_l2"] <= 1e-6,
                   "baseline": baseline_time, "candidate": candidate_time,
                   "speedup": baseline_time["median_ms"] / candidate_time["median_ms"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows

def _main():
    """Select an idle device explicitly and persist machine-readable results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=("ns", "momentum", "writeback", "apply", "gram"), default="ns")
    parser.add_argument("--gram-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    with torch.no_grad():
        runners = {"ns": _ns_cases, "gram": _ns_cases, "momentum": _momentum_cases,
                   "writeback": _pointwise_cases, "apply": _pointwise_cases}
        rows = runners[args.scheme](args)
    report = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "rows": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _main()

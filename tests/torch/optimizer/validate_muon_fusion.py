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

"""Validate independent Muon switches with real NPU optimizer trajectories.

Run with torchrun. The distributed cases reuse the existing Muon state and
parameter self-consistency contract instead of weakening it for optimizations.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.optimizer.muon import Muon
from tests.torch.optimizer._test_muon_e2e import (
    _assert_matches_reference,
    _build_model,
    _make_optimizer,
    _run_reference_trajectory,
)
from tests.torch.optimizer.benchmark_muon_fusion import _errors, _timing
from tests.torch.utils import init_dist


_OPTIONS = {
    "ns": {"ns_implementation": "standard_fused"},
    "momentum": {"momentum_implementation": "triton"},
    "writeback": {"fused_scale_writeback": True},
    "apply": {"update_implementation": "triton"},
    "gram": {"ns_implementation": "gram"},
}


def _plain_case(options, nesterov):
    torch.manual_seed(28)
    params = [torch.nn.Parameter(torch.randn(shape, device="npu") * .02)
              for shape in ((64, 128), (128, 64), (3, 32, 64))]
    actual = [torch.nn.Parameter(param.detach().clone()) for param in params]
    kwargs = {"lr": .01, "momentum": (.95, .85), "nesterov": nesterov}
    reference = Muon(params, **kwargs)
    candidate = Muon(actual, **kwargs, **options)
    max_error = 0.0
    max_state_error = 0.0
    close = True
    state_close = True
    for step in range(30):
        torch.manual_seed(2026 + step)
        for param, target in zip(params, actual):
            grad = torch.randn_like(param) * .05
            param.grad = grad
            target.grad = grad.clone()
        reference.step()
        candidate.step()
        for param, target in zip(params, actual):
            max_error = max(max_error, _errors(target, param)["relative_l2"])
            close &= torch.allclose(target, param, rtol=1e-2, atol=1e-3)
            expected_buf = reference.state[param]["momentum_buffer"]
            actual_buf = candidate.state[target]["momentum_buffer"]
            max_state_error = max(max_state_error, _errors(actual_buf, expected_buf)["relative_l2"])
            state_close &= torch.allclose(actual_buf, expected_buf, rtol=1e-4, atol=1e-6)
    restored_params = [torch.nn.Parameter(param.detach().clone()) for param in actual]
    restored = Muon(restored_params, **kwargs, **options)
    restored.load_state_dict(copy.deepcopy(candidate.state_dict()))
    for source, target in zip(actual, restored_params):
        target.grad = source.grad.clone()
    candidate.step()
    restored.step()
    for source, target in zip(actual, restored_params):
        torch.testing.assert_close(source, target, rtol=0, atol=0)
    baseline_time = _timing(reference.step, 20)
    candidate_time = _timing(candidate.step, 20)
    return {"baseline": baseline_time, "candidate": candidate_time,
            "speedup": baseline_time["median_ms"] / candidate_time["median_ms"],
            "nesterov": nesterov, "steps": 30, "parameter_relative_l2": max_error,
            "state_relative_l2": max_state_error, "parameter_close": bool(close),
            "state_close": bool(state_close), "checkpoint_exact": True}


def _shard(model):
    mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("dp",))
    return fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy())



def _training_case(options):
    torch.manual_seed(29)
    baseline = torch.nn.Sequential(torch.nn.Linear(64, 128, bias=False), torch.nn.Tanh(),
                                   torch.nn.Linear(128, 32, bias=False)).npu()
    candidate = copy.deepcopy(baseline)
    optimizers = [Muon(baseline.parameters(), lr=.005), Muon(candidate.parameters(), lr=.005, **options)]
    inputs = torch.randn(16, 64, device="npu")
    targets = torch.randn(16, 32, device="npu")
    losses = [[], []]
    for _ in range(20):
        for index, (model, optimizer) in enumerate(zip((baseline, candidate), optimizers)):
            optimizer.zero_grad()
            loss = (model(inputs) - targets).square().mean()
            loss.backward()
            optimizer.step()
            losses[index].append(loss.item())
    parameter_errors = [_errors(right, left) for left, right in zip(baseline.parameters(), candidate.parameters())]
    return {"baseline_losses": losses[0], "candidate_losses": losses[1], "parameter_errors": parameter_errors,
            "loss_close": torch.allclose(torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=.01, atol=1e-4),
            "parameter_close": all(torch.allclose(left, right, rtol=.01, atol=.001)
                                   for left, right in zip(baseline.parameters(), candidate.parameters()))}

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--scheme", choices=tuple(_OPTIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    init_dist()
    options = _OPTIONS[args.scheme]
    report = {"scheme": args.scheme, "world_size": dist.get_world_size(), "plain": []}
    if not args.training_only:
        with torch.no_grad():
            for nesterov in (False, True):
                report["plain"].append(_plain_case(options, nesterov))
            try:
                reference = _run_reference_trajectory()
                model = _shard(_build_model())
                optimizer = _make_optimizer(model, **options)
                _assert_matches_reference(model, optimizer, reference)
                report["sharded_parity"] = True
            except AssertionError as exc:
                report["sharded_parity"] = False
                report["sharded_error"] = str(exc)
    report["training"] = _training_case(options)
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()
    if args.require_pass:
        plain_pass = all(row["parameter_close"] and row["state_close"] and row["checkpoint_exact"]
                         for row in report["plain"])
        passed = (plain_pass and report.get("sharded_parity", True)
                  and report["training"]["loss_close"] and report["training"]["parameter_close"])
        if not passed:
            raise AssertionError(f"Muon {args.scheme} did not pass the unchanged accuracy contract")


if __name__ == "__main__":
    _main()

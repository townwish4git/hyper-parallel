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
"""Compare the production Muon use_symm_gemm flag on full Qwen model specifications."""

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu

import hyper_parallel
from hyper_parallel import init_device_mesh
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.muon import Muon
from tests.torch.optimizer.muon_model_inputs import (
    MODEL_NUMEL, build_model, initialize_gradients, initialize_parameters, model_config_path,
)
from tests.torch.optimizer.muon_step_metrics import empty_metrics, merge_metrics, summarize_metrics, tensor_metrics


_METRICS = ("parameters", "effective_updates", "momentum")


def _options(enabled):
    return {"use_symm_gemm": enabled}


def _optimizer(parameters, enabled):
    optimizer = Muon(parameters, **_options(enabled), lr=0.01, weight_decay=0.1,
                     momentum=0.95, nesterov=True, ns_steps=5, ns_variant="asym5")
    if optimizer.zeropower_fn is not None or optimizer.momentum_update_fn is not None:
        raise AssertionError("Only the built-in optimizer is allowed")
    return optimizer


def _reset(parameters, seed, enabled):
    initialize_parameters(parameters, seed)
    initialize_gradients(parameters, seed)
    optimizer = _optimizer(parameters, enabled)
    torch.npu.synchronize()
    dist.barrier()
    return optimizer


def _release():
    gc.collect()
    torch.npu.empty_cache()
    dist.barrier()


def _timing(parameters, enabled, args):
    optimizer = _reset(parameters, args.seed, enabled)
    for _ in range(args.warmup):
        optimizer.step()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    samples = []
    for _ in range(args.steps):
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        optimizer.step()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        values = torch.tensor(elapsed, device="npu", dtype=torch.float32)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        samples.append(values.item())
    memory = torch.tensor(torch.npu.max_memory_allocated(), device="npu", dtype=torch.int64)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    del optimizer
    _release()
    return {"enabled": enabled, **_options(enabled), "slowest_rank_ms": samples,
            "median_ms": statistics.median(samples),
            "max_peak_allocated_gib": memory.item() / 2**30}


def _snapshot(parameters, optimizer):
    return [(to_local_if_dtensor(p).cpu(),
             to_local_if_dtensor(optimizer.state[p]["momentum_buffer"]).cpu()
             if "momentum_buffer" in optimizer.state[p] else None) for p in parameters]


def _compare(parameters, names, optimizer, initial, baseline):
    rows = []
    for parameter, name, origin, (reference, momentum) in zip(parameters, names, initial, baseline):
        actual = to_local_if_dtensor(parameter).cpu()
        state = optimizer.state[parameter].get("momentum_buffer")
        if (state is None) != (momentum is None):
            raise AssertionError(f"Momentum ownership differs: {name}")
        row = {"name": name, "shape": list(parameter.shape),
               "parameters": tensor_metrics(actual, reference),
               "effective_updates": tensor_metrics(actual, reference, origin),
               "momentum": empty_metrics() if state is None else tensor_metrics(
                   to_local_if_dtensor(state).cpu(), momentum)}
        if any(row[key]["nonfinite"] for key in _METRICS):
            raise AssertionError(f"Nonfinite output: {name}")
        rows.append(row)
    return rows


def _precision_summary(ranks):
    merged = {}
    for rows in ranks:
        for row in rows:
            target = merged.setdefault(row["name"], {"name": row["name"], "shape": row["shape"],
                                                     **{key: empty_metrics() for key in _METRICS}})
            for key in _METRICS:
                merge_metrics(target[key], row[key])
    totals = {key: empty_metrics() for key in _METRICS}
    groups = {}
    parameters = []
    for row in merged.values():
        shape = tuple(row["shape"])
        group = groups.setdefault(shape, {"shape": shape, "parameter_count": 0,
                                          **{key: empty_metrics() for key in _METRICS}})
        group["parameter_count"] += 1
        for key in _METRICS:
            merge_metrics(totals[key], row[key])
            merge_metrics(group[key], row[key])
        parameters.append({"name": row["name"], "shape": row["shape"],
                           **{key: summarize_metrics(row[key]) for key in _METRICS}})
    by_shape = [{"shape": group["shape"], "parameter_count": group["parameter_count"],
                 **{key: summarize_metrics(group[key]) for key in _METRICS}} for group in groups.values()]
    return {"global": {key: summarize_metrics(value) for key, value in totals.items()}, "by_shape": by_shape,
            "worst_update_parameters": sorted(parameters, key=lambda r: r["effective_updates"]["relative_l2"],
                                              reverse=True)[:10]}


def _accuracy(parameters, names, args, steps):
    initialize_parameters(parameters, args.seed)
    initial = [to_local_if_dtensor(p).cpu() for p in parameters]
    baseline_mode = False
    candidates = [True]
    optimizer = _reset(parameters, args.seed, baseline_mode)
    for step in range(steps):
        initialize_gradients(parameters, args.seed, step)
        optimizer.step()
    torch.npu.synchronize()
    baseline = _snapshot(parameters, optimizer)
    del optimizer
    _release()
    results = []
    for mode in candidates:
        optimizer = _reset(parameters, args.seed, mode)
        for step in range(steps):
            initialize_gradients(parameters, args.seed, step)
            optimizer.step()
        torch.npu.synchronize()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(args.metric_threads)
            local = _compare(parameters, names, optimizer, initial, baseline)
        finally:
            torch.set_num_threads(previous_threads)
        del optimizer
        _release()
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, local)
        result = _precision_summary(ranks)
        result.update(steps=steps, mode=mode)
        if result["global"]["momentum"]["different_bit_patterns"]:
            raise AssertionError("The configuration changed momentum state")
        results.append(result)
        if dist.get_rank() == 0:
            print(json.dumps({"precision_steps": steps, "mode": mode, "global": result["global"]}), flush=True)
    del baseline, initial
    return results


def _write(report, output):
    if dist.get_rank() == 0:
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _source_hashes():
    root = Path(hyper_parallel.__file__).parent
    names = ("core/optimizer/muon.py", "components/functional/symmetric_matmul.py",
             "components/functional/_ascendc/symmetric_matmul/launch.py",
             "components/functional/_ascendc/symmetric_matmul/libhp_symmetric.so",
             "components/functional/_ascendc/symmetric_matmul/libhp_fused_kernel.so",
             "components/functional/_ascendc/symmetric_matmul/libhp_epilogue_kernel.so")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


@torch.no_grad()
def main() -> None:
    """Benchmark only the public production flag, with separate full-element precision runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_NUMEL), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=0, help="Nonzero is diagnostic only, not a full-model result")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=51430)
    parser.add_argument("--accuracy-steps", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--metric-threads", type=int, default=1,
                        help="CPU threads only for precision statistics; "
                             "optimizer execution keeps its original setting")
    args = parser.parse_args()
    if args.metric_threads < 1:
        raise ValueError("--metric-threads must be positive")
    if os.environ.get("CLOSE_MATMUL_K_SHIFT") != "1":
        raise ValueError("The precision comparison requires CLOSE_MATMUL_K_SHIFT=1 before process startup")
    modes = [False, True]
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=30))
    mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("shard",))
    model, parameters, names, description = build_model(args.model, mesh, args.layers)
    report = {"args": vars(args), "host": socket.gethostname(), "world_size": dist.get_world_size(),
              "device": torch.npu.get_device_name(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "source": hyper_parallel.__file__, "source_hashes": _source_hashes(), "model": description,
              "config_sha256": hashlib.sha256(model_config_path(args.model).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "comparison": {str(mode): _options(mode) for mode in modes},
              "close_matmul_k_shift": os.environ.get("CLOSE_MATMUL_K_SHIFT"),
              "scope": "Muon.step only: communication, momentum and updates; "
                       "no AdamW, forward/backward or grad reduction.",
              "precision_scope": "All Muon parameter shards and momentum, not samples. "
                                 "Effective update = final - initial.",
              "protocol": "BF16 params/gradients/state; no FP32 master. Timings use fixed gradients; "
                          "precision uses identical changing gradients. Fresh state and same initial values per run.",
              "blocks": [], "precision": []}
    _write(report, args.output)
    print(json.dumps({"ready": description["full_numel"], "rank": dist.get_rank()}), flush=True)
    for cycle in range(args.cycles):
        for enabled in (modes if cycle % 2 == 0 else list(reversed(modes))):
            row = _timing(parameters, enabled, args)
            row["cycle"] = cycle
            report["blocks"].append(row)
            _write(report, args.output)
            if dist.get_rank() == 0:
                print(json.dumps(row), flush=True)
    for steps in args.accuracy_steps:
        rows = _accuracy(parameters, names, args, steps)
        report["precision"].extend(rows)
        _write(report, args.output)
    del model
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

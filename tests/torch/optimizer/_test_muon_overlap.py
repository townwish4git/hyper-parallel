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
"""Muon pipeline parity and optimizer-only benchmarks; execute using torchrun."""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from hyper_parallel.core.optimizer.muon import Muon


def _build_pair(args, mesh):
    torch.manual_seed(123)
    serial_params, pipeline_params = [], []
    for index in range(args.count):
        full = torch.randn(args.rows, args.cols, device="npu") * 0.02
        grad = torch.randn_like(full) * 0.05
        for params in (serial_params, pipeline_params):
            param = torch.nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
            param.model_name = f"matrix_{index}"
            param.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
            params.append(param)
    for index in range(args.local_count):
        full = torch.randn(dist.get_world_size() * 2, 64, 32, device="npu") * 0.02
        grad = torch.randn_like(full) * 0.05
        for params in (serial_params, pipeline_params):
            param = torch.nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
            param.model_name = f"experts_{index}"
            if index != args.local_count - 1:
                param.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
            params.append(param)
    kwargs = {"lr": 0.01, "weight_decay": 0.1, "fsdp_overlap_memory_budget_bytes": args.budget}
    return Muon(serial_params, **kwargs), Muon(pipeline_params, fsdp_overlap=True, **kwargs)


def _check_pair(serial, pipeline):
    for reference, actual in zip(serial.param_groups[0]["params"], pipeline.param_groups[0]["params"]):
        torch.testing.assert_close(actual.to_local(), reference.to_local(), rtol=0, atol=0)
        if reference.grad is None:
            if actual in pipeline.state or reference in serial.state:
                raise AssertionError("A missing gradient must not create momentum state")
            continue
        torch.testing.assert_close(
            pipeline.state[actual]["momentum_buffer"].to_local(),
            serial.state[reference]["momentum_buffer"].to_local(), rtol=0, atol=0,
        )


def _measure(optimizer, args):
    for _ in range(args.warmup):
        optimizer.step()
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.reset_peak_memory_stats()
    times = []
    for _ in range(args.steps):
        start = time.perf_counter()
        optimizer.step()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(times), "samples_ms": times,
            "peak_allocated_bytes": torch.npu.max_memory_allocated()}


def main() -> None:
    """Compare identical sharded inputs, then measure ABBA optimizer timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--local-count", type=int, default=0)
    parser.add_argument("--budget", type=int, default=80 * 1024 * 1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    mesh = init_device_mesh("npu", (dist.get_world_size(),))
    serial, pipeline = _build_pair(args, mesh)
    assignment = pipeline.shard_assignments_by_group[0][0]
    batches = pipeline._shard_pipeline_batches(assignment.owned_params, dist.get_world_size())
    oversized = any(pipeline._pipeline_parameter_bytes(param, dist.get_world_size()) > args.budget // 2
                    for param in assignment.owned_params)
    if len(batches) < 2 or oversized or not pipeline._can_overlap_shard(assignment):
        raise AssertionError("Benchmark must exercise the pipeline with at least two buckets")
    pipeline_calls = [0]
    parity_bucket_counts = []
    enqueue_ns = pipeline._enqueue_pipeline_ns

    def _count_pipeline_ns(*positional, **keywords):
        pipeline_calls[0] += 1
        return enqueue_ns(*positional, **keywords)

    pipeline._enqueue_pipeline_ns = _count_pipeline_ns
    caller = torch.npu.Stream()
    caller.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(caller):
        for step in range(4):
            serial.step()
            pipeline_calls[0] = 0
            pipeline.step()
            if pipeline_calls[0] != len(batches):
                raise AssertionError("Parity test fell back instead of executing every pipeline bucket")
            parity_bucket_counts.append(pipeline_calls[0])
            _check_pair(serial, pipeline)
            if step == 1:
                pipeline.load_state_dict(copy.deepcopy(pipeline.state_dict()))
                pipeline.reset_optimizer_parameters()
    torch.npu.current_stream().wait_stream(caller)
    results = []
    for label, optimizer in (("serial", serial), ("pipeline", pipeline),
                             ("pipeline", pipeline), ("serial", serial)):
        results.append({"mode": label, **_measure(optimizer, args)})
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, results)
    if dist.get_rank() == 0:
        summary = {"args": vars(args), "world_size": dist.get_world_size(),
                   "bucket_sizes": [len(batch) for batch in batches], "parity": "bitwise_pass",
                   "parity_pipeline_bucket_counts": parity_bucket_counts, "ranks": gathered}
        print(json.dumps(summary), flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.profile:
        for label, optimizer in (("serial", serial), ("pipeline", pipeline)):
            schedule = torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1)
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=schedule) as prof:
                for _ in range(4):
                    optimizer.step()
                    torch.npu.synchronize()
                    prof.step()
            if dist.get_rank() == 0:
                prof.export_chrome_trace(f"{args.profile}_{label}.json")
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

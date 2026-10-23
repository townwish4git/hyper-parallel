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

"""Unit tests for the Muon optimizer."""

from collections import defaultdict, deque
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer


def _plain_ns_group(**overrides):
    """Group dict covering every key consumed by the batched NS paths."""
    group = {
        "lr": 0.01,
        "matched_adamw_rms": 0.2,
        "ns_steps": 1,
        "ns_variant": "asym5",
        "ns_coefficients": None,
        "ns_epsilon": 1e-10,
        "zero_rms_scale_mode": "zero",
        "apply_lr_in_update": False,
    }
    group.update(overrides)
    return group


def test_compute_batched_ns_updates_accepts_plain_parameter_without_sharding():
    """The no-shard path should support ordinary torch parameters."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = None
    optimizer.zeropower_fn = None
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    ns_inputs = {parameter: torch.ones(2, 2)}
    group = _plain_ns_group()

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], ns_inputs, group, no_shard=True)

    assert updates[parameter].shape == parameter.shape


def test_compute_batched_ns_updates_with_transform_accepts_plain_parameter():
    """The ns_transform_fn routing path should also support plain parameters."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = lambda param_fqn, ns_input: None
    optimizer.zeropower_fn = None
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    ns_inputs = {parameter: torch.ones(2, 2)}
    group = _plain_ns_group()

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], ns_inputs, group, no_shard=True)

    assert updates[parameter].shape == parameter.shape


def test_muon_rejects_invalid_ns_variant():
    """ns_variant outside legacy/asym5/custom must raise."""
    with pytest.raises(ValueError, match="ns_variant"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_variant="unknown")


def test_muon_rejects_ns_coefficients_without_custom_variant():
    """ns_coefficients is only valid together with ns_variant='custom'."""
    with pytest.raises(ValueError, match="ns_coefficients"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="asym5",
            ns_coefficients=[(3.4445, -4.7750, 2.0315)],
        )


def test_muon_rejects_missing_or_short_ns_coefficients():
    """Custom NS requires at least ns_steps coefficient groups."""
    with pytest.raises(ValueError, match="ns_coefficients"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_variant="custom")
    with pytest.raises(ValueError, match="exceeds"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=2,
            ns_coefficients=[(3.4445, -4.7750, 2.0315)],
        )


def test_muon_rejects_malformed_ns_coefficients():
    """Each custom coefficient group must hold three finite values."""
    with pytest.raises(ValueError, match="exactly three"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=1,
            ns_coefficients=[(3.4445, -4.7750)],
        )
    with pytest.raises(ValueError, match="finite"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=1,
            ns_coefficients=[(3.4445, float("inf"), 2.0315)],
        )


def test_muon_rejects_invalid_ns_epsilon_and_zero_rms_scale_mode():
    """ns_epsilon must be finite non-negative; zero_rms_scale_mode must be known."""
    with pytest.raises(ValueError, match="ns_epsilon"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_epsilon=-1e-10)
    with pytest.raises(ValueError, match="zero_rms_scale_mode"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], zero_rms_scale_mode="keep")


def test_async_hierarchical_broadcast_waits_between_mesh_dimensions():
    """A relay dimension must wait before forwarding the shared buffer."""
    first_work = MagicMock()
    final_work = MagicMock()
    process_groups = (MagicMock(), MagicMock())

    with patch(
            "hyper_parallel.core.optimizer.optimizer.dist.get_global_rank",
            side_effect=[0, 1],
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist.broadcast",
            side_effect=[first_work, final_work],
    ) as broadcast:
        handles = BaseDistributedOptimizer._hierarchical_broadcast_buffer_async(  # pylint: disable=protected-access
            torch.ones(4),
            src_coord=(0, 0),
            replicate_pgs=process_groups,
            local_coord=(0, 0),
        )

    assert broadcast.call_count == 2
    first_work.wait.assert_called_once_with()
    final_work.wait.assert_not_called()
    assert handles == [final_work]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_muon_overlap_rejects_invalid_budget(budget):
    """A workspace budget must be an explicit positive byte count."""
    with pytest.raises(ValueError, match="positive integer"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], fsdp_overlap_memory_budget_bytes=budget)


def test_muon_overlap_rejects_non_boolean_switch():
    """Reject accidental string configuration before distributed initialization."""
    with pytest.raises(TypeError, match="bool"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], fsdp_overlap="true")


@pytest.mark.parametrize("bucket_count", [2, 3, 5])
@pytest.mark.parametrize("queue_capacity", [2, 4])
def test_muon_pipeline_collective_order_and_backpressure(bucket_count, queue_capacity):
    """Submit broadcast before host backpressure and prepare momentum only once."""
    optimizer = Muon.__new__(Muon)
    optimizer._overlap_streams = {}
    optimizer.fsdp_overlap_memory_budget_bytes = queue_capacity
    optimizer._shard_compute_coord = {0: {}}
    params = [MagicMock() for _ in range(bucket_count)]
    optimizer._param_shard_metadata = {p: MagicMock() for p in params}
    for meta in optimizer._param_shard_metadata.values():
        meta.stage_metas[0].pad_shape = (1, 1)
    buckets = [{"index": i, "apply_done": MagicMock()} for i in range(bucket_count)]
    assignment = MagicMock(owned_params=params, shard_pgs=(MagicMock(),))
    backend = MagicMock()
    events = []

    def _gather(bucket, *_args):
        events.append(("gather", bucket["index"]))

    def _record_ns(bucket, *_args):
        events.append(("ns", bucket["index"]))

    def _record_broadcast(bucket, *_args):
        events.append(("broadcast", bucket["index"]))

    def _record_apply(bucket, *_args):
        events.append(("apply", bucket["index"]))

    def _retire(bucket):
        if bucket is not None:
            events.append(("retire", bucket["index"]))

    with patch.object(optimizer, "_pipeline_bucket_bytes", return_value=1), \
            patch.object(optimizer, "_pipeline_prepared_bytes", return_value=0), \
            patch.object(optimizer, "_shard_pipeline_batches", return_value=[[p] for p in params]), \
            patch.object(optimizer, "_update_muon_momentum", return_value={}) as momentum, \
            patch.object(optimizer, "_prepare_pipeline_buckets", return_value=buckets), \
            patch.object(optimizer, "_launch_pipeline_gather", side_effect=_gather), \
            patch.object(optimizer, "_enqueue_pipeline_ns", side_effect=_record_ns), \
            patch.object(optimizer, "_enqueue_pipeline_broadcast", side_effect=_record_broadcast), \
            patch.object(optimizer, "_enqueue_pipeline_apply", side_effect=_record_apply), \
            patch.object(optimizer, "_retire_pipeline_bucket", side_effect=_retire), \
            patch("hyper_parallel.core.optimizer.muon.to_local_if_dtensor",
                  return_value=MagicMock(device=torch.device("cuda:0"))), \
            patch("hyper_parallel.core.optimizer.muon.dist.get_world_size", return_value=2), \
            patch("hyper_parallel.core.optimizer.muon.torch.cuda", backend):
        optimizer._process_shard_params_pipelined({}, assignment, 0)

    collectives = [event for event in events if event[0] in ("gather", "broadcast")]
    expected = [("gather", 0)]
    for index in range(bucket_count):
        if index + 1 < bucket_count:
            expected.append(("gather", index + 1))
        expected.append(("broadcast", index))
    assert collectives == expected
    momentum.assert_called_once_with({}, params)
    for index in range(1, bucket_count):
        assert events.index(("broadcast", index - 1)) < events.index(("ns", index))
        assert events.index(("ns", index)) < events.index(("apply", index - 1))
    queue_depth = min(bucket_count, queue_capacity)
    retired = [event for event in events if event[0] == "retire"]
    assert retired == [("retire", index) for index in range(bucket_count - queue_depth)]
    for _, index in retired:
        assert events.index(("broadcast", index + queue_depth - 1)) < events.index(("retire", index))
    assert events[-1] == ("apply", bucket_count - 1)


def test_muon_pipeline_preserves_replica_and_callback_fallbacks():
    """HSDP and custom callbacks retain the original stream semantics."""
    optimizer = Muon.__new__(Muon)
    optimizer.fsdp_overlap = True
    assignment = MagicMock(is_shard=True, is_replicated=True)
    assert not optimizer._can_overlap_shard(assignment)
    assignment.is_replicated = False
    assignment.shard_pgs = (MagicMock(),)
    for name in ("reshape_fn", "zeropower_fn", "momentum_update_fn", "ns_transform_fn", "post_update_fn"):
        setattr(optimizer, name, None)
    optimizer.post_update_fn = MagicMock()
    assert not optimizer._can_overlap_shard(assignment)


def test_muon_overlap_configuration_rejects_rank_disagreement():
    """Detect a divergent collective schedule before the first optimizer step."""
    optimizer = Muon.__new__(Muon)
    optimizer.fsdp_overlap = True
    optimizer.fsdp_overlap_memory_budget_bytes = 4096
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    assignment = MagicMock(is_replicated=False, shard_pgs=(MagicMock(),), owned_params=[parameter])
    optimizer.shard_assignments_by_group = {0: [assignment]}

    def _gather(output, config, **_kwargs):
        output.view(2, 3).copy_(config.expand(2, 3))
        output[3] = 0

    with patch.object(optimizer, "_can_overlap_shard", return_value=True), \
            patch("hyper_parallel.core.optimizer.muon.dist.get_world_size", return_value=2), \
            patch("hyper_parallel.core.optimizer.muon.dist.all_gather_into_tensor", side_effect=_gather):
        with pytest.raises(ValueError, match="must agree across ranks"):
            optimizer._validate_overlap_configuration()


def test_muon_overlap_budget_preserves_serial_default_and_bounds_buckets():
    """An implicit overlap budget must not change unsupported serial/HSDP execution."""
    optimizer = Muon.__new__(Muon)
    optimizer.fsdp_overlap_memory_budget_bytes = None
    params = [object() for _ in range(4)]
    with patch.object(optimizer, "_split_into_memory_safe_batches", return_value=[params]), \
            patch.object(optimizer, "_pipeline_prepared_bytes", return_value=0), \
            patch.object(optimizer, "_pipeline_parameter_bytes", return_value=128 * 1024 * 1024):
        assert optimizer._shard_pipeline_batches(params, 2) == [params]
        assert optimizer._shard_pipeline_batches(params, 2, use_overlap_budget=True) == [params[:2], params[2:]]


def test_muon_gather_reuse_does_not_wait_for_parameter_apply():
    """The receive slot is reusable once its old NS consumer has finished."""
    bucket = {"prepared": MagicMock()}
    previous = {"ns_done": MagicMock(), "apply_done": MagicMock()}
    stream = MagicMock()
    assignment = MagicMock(shard_pgs=[MagicMock()])
    cache = {}
    Muon._launch_pipeline_gather(bucket, assignment, cache, previous, stream)
    stream.wait_event.assert_called_once_with(previous["ns_done"])
    previous["apply_done"].synchronize.assert_not_called()
    bucket["prepared"].launch.assert_called_once_with(assignment.shard_pgs[0], cache)


def test_muon_pipeline_backpressure_does_not_wait_for_apply():
    """Host throttling bounds NS temporaries without draining the update stream."""
    bucket = {"ns_done": MagicMock(), "apply_done": MagicMock()}
    Muon._retire_pipeline_bucket(bucket)
    bucket["ns_done"].synchronize.assert_called_once_with()
    bucket["apply_done"].synchronize.assert_not_called()


def test_muon_bucket_workspace_is_owned_per_rank():
    """Eight owners need one matrix of NS workspace each, not eight on every rank."""
    optimizer = Muon.__new__(Muon)
    params = [torch.nn.Parameter(torch.empty(512, 1024, device="meta")) for _ in range(8)]
    optimizer._shard_param_compute_coord = {param: (rank,) for rank, param in enumerate(params)}
    optimizer._param_shard_metadata = {param: MagicMock() for param in params}
    for meta in optimizer._param_shard_metadata.values():
        meta.stage_metas[0].pad_shape = (64, 1024)
    optimizer.fsdp_overlap_memory_budget_bytes = 80 * 1024 * 1024
    with patch.object(optimizer, "_split_into_memory_safe_batches", return_value=[params]):
        assert optimizer._shard_pipeline_batches(params, 8) == [params]
    distributed = optimizer._pipeline_bucket_bytes(params, 8)
    optimizer._shard_param_compute_coord = {param: (0,) for param in params}
    concentrated = optimizer._pipeline_bucket_bytes(params, 8)
    assert concentrated - distributed == 7 * 32 * params[0].numel()


def test_muon_local_ns_batches_preserve_grouping_and_skip_empty_shards():
    """Interleaving must preserve the original NS batches and consume each input once."""
    optimizer = Muon.__new__(Muon)
    params = [torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(3, 2)),
              torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(0, 3))]
    inputs = {param: torch.zeros_like(param) for param in params}
    batches = list(optimizer._unshard_ns_batches(inputs))
    assert [list(batch) for batch in batches] == [[params[0], params[2]], [params[1]]]
    for batch in batches:
        for param, value in batch.items():
            assert value is inputs[param]


def test_muon_interleaved_local_updates_match_unsplit_execution():
    """Moving complete native batches must preserve updates including weight decay."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = None
    optimizer.zeropower_fn = None
    torch.manual_seed(17)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in [(2, 3), (3, 2), (2, 3)]]
    inputs = {param: torch.randn_like(param, dtype=torch.bfloat16) for param in params}
    original_inputs = {param: value.clone() for param, value in inputs.items()}
    originals = [param.detach().clone() for param in params]
    group = _plain_ns_group(weight_decay=0.1)
    with torch.no_grad():
        optimizer._process_unshard_params(group, inputs)
        expected = [param.clone() for param in params]
        for param, original in zip(params, originals):
            param.copy_(original)
            inputs[param].copy_(original_inputs[param])
        batches = list(optimizer._unshard_ns_batches(inputs))
        for batch in batches:
            optimizer._process_unshard_params(group, batch)
    for param, reference in zip(params, expected):
        torch.testing.assert_close(param, reference, rtol=0, atol=0)


def test_muon_lazy_local_momentum_matches_eager_updates():
    """Local assignment inputs may be generated lazily without changing momentum or updates."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = None
    optimizer.zeropower_fn = None
    optimizer.momentum_update_fn = None
    optimizer.state = defaultdict(dict)
    torch.manual_seed(23)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in [(2, 3), (3, 2)]]
    for param in params:
        param.grad = torch.randn_like(param)
    initial = [param.detach().clone() for param in params]
    group = _plain_ns_group(weight_decay=0.1, momentum=0.95, nesterov=True)
    with torch.no_grad():
        inputs = optimizer._update_muon_momentum(group, params)
        optimizer._process_unshard_params(group, inputs)
        expected = [param.clone() for param in params]
        buffers = [optimizer.state[param]["momentum_buffer"].clone() for param in params]
        optimizer.state.clear()
        for param, value in zip(params, initial):
            param.copy_(value)
        work = {"batches": deque(([param], None) for param in params)}
        for _ in params:
            optimizer._run_local_ns_batch(group, work)
        optimizer._run_local_ns_batch(group, work)
    assert not work["batches"]
    for param, value, buffer in zip(params, expected, buffers):
        torch.testing.assert_close(param, value, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"], buffer, rtol=0, atol=0)

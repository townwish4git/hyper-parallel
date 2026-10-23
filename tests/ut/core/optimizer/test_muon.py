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

from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer
from hyper_parallel.core.optimizer.sharding_category import (
    CommDomainKey,
    HSDPCommGroup,
    ParamLayoutSpec,
    ParamRecord,
)
from tests.common.mark_utils import arg_mark


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


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
def test_muon_rejects_non_boolean_hsdp_replicate_dedup(value):
    """
    Feature: Muon replicate dedup configuration validation.
    Description: Pass non-boolean values to the public constructor.
    Expectation: Invalid values raise ValueError.
    """
    with pytest.raises(ValueError, match="hsdp_replicate_dedup must be a bool"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], hsdp_replicate_dedup=value)


def _make_replica_optimizer(replicate_sizes, replicate_coord, is_shard=False, **options):
    """Use real assignment construction with mocked mesh topology and collectives."""
    params = [torch.nn.Parameter(torch.full((2, 2), float(index + 1))) for index in range(6)]
    for param in params:
        param.device_mesh = MagicMock()
    replicate_dims = tuple(range(len(replicate_sizes)))
    shard_axes = ((len(replicate_sizes), 0),) if is_shard else ()
    layout = ParamLayoutSpec(2, shard_axes, replicate_dims)
    comm_key = CommDomainKey(
        replicate_sizes + ((2,) if is_shard else ()), (), replicate_dims, layout.shard_mesh_dims,
    )
    hsdp_group = HSDPCommGroup(
        comm_key, layout,
        replicate_pgs=tuple(MagicMock() for _ in replicate_sizes),
        shard_pgs=(MagicMock(),) if is_shard else (),
        records=[ParamRecord(index, param) for index, param in enumerate(params)],
    )
    with patch(
            "hyper_parallel.core.optimizer.optimizer.group_parameters_for_hsdp",
            return_value=([], [hsdp_group]),
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.get_multi_dim_logical_info",
            return_value=(replicate_coord, replicate_sizes),
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist",
    ) as mock_dist, patch(
            "hyper_parallel.core.optimizer.sharding_category.dist",
    ), patch.object(
            Muon, "_auto_deduce_replica_count", return_value=None,
    ), patch.object(
            Muon, "_build_param_shard_metadata",
    ), patch.object(
            Muon, "_get_shard_info", return_value=((2,), (0,), hsdp_group.shard_pgs, 2),
    ):
        optimizer = Muon(params, **options)
        # Parameter replacement / checkpoint restores also rebuild these caches.
        optimizer.reset_optimizer_parameters()
        if options.get("hsdp_replicate_dedup") is False:
            mock_dist.new_group.assert_not_called()
    return optimizer, params


@pytest.mark.parametrize("replicate_sizes, replicate_coord", [((2,), (1,)), ((2, 3), (1, 2))])
@pytest.mark.parametrize("is_shard", [False, True])
@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
def test_muon_disabled_dedup_owns_all_params_without_replica_broadcast(replicate_sizes, replicate_coord, is_shard):
    """
    Feature: Disable Muon replicate dedup on single- and multi-axis meshes.
    Description: Build and reset assignments on a nonzero replica with and without sharding.
    Expectation: Every replica owns all parameters, with shard communication preserved and no replica broadcasts.
    """
    optimizer, params = _make_replica_optimizer(
        replicate_sizes, replicate_coord, is_shard,
        hsdp_replicate_dedup=False, hsdp_replica_count=2,
    )
    assignment = optimizer._hsdp_assignment_batches[0]["batch_groups"][0]["sub_batches"][0]
    assert [id(param) for param in assignment.owned_params] == [id(param) for param in params], (
        f"owned param count={len(assignment.owned_params)}, expected={len(params)}"
    )
    assert not assignment.is_replicated, f"effective replica sizes={assignment.replicate_sizes}"
    assert assignment.is_shard == is_shard, f"is_shard={assignment.is_shard}, expected={is_shard}"
    assert bool(assignment.shard_pgs) == is_shard, f"shard groups={assignment.shard_pgs}, expected={is_shard}"
    assert not optimizer._param_to_broadcast_info, f"unexpected broadcasts={optimizer._param_to_broadcast_info}"


@pytest.mark.parametrize("options", [{}, {"hsdp_replicate_dedup": True}])
@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
def test_muon_default_and_enabled_dedup_partition_replica_ownership(options):
    """
    Feature: Preserve Muon replicate dedup by default.
    Description: Compare ownership on two replicas using default and explicit enabled settings.
    Expectation: Ownership partitions the parameters and registers replicate broadcasts.
    """
    owned_indices = []
    for rank in range(2):
        optimizer, _ = _make_replica_optimizer((2,), (rank,), **options)
        assignment = optimizer._hsdp_assignment_batches[0]["batch_groups"][0]["sub_batches"][0]
        owned_indices.append({record.index for record in assignment.owned_records})
        assert assignment.is_replicated, f"effective replica sizes={assignment.replicate_sizes}"
        assert len(optimizer._param_to_broadcast_info) == 6, (
            f"broadcast count={len(optimizer._param_to_broadcast_info)}, expected=6"
        )
    assert owned_indices[0].isdisjoint(owned_indices[1]), f"overlapping owners={owned_indices}"
    assert owned_indices[0] | owned_indices[1] == set(range(6)), f"incomplete owners={owned_indices}"


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
def test_muon_disabled_dedup_step_matches_plain_muon_and_keeps_momentum():
    """
    Feature: Muon numerical behavior with replicate dedup disabled.
    Description: Run repeated steps on a nonzero replica and a plain optimizer.
    Expectation: Parameters and momentum match at every step without replica broadcasts.
    """
    optimizer, params = _make_replica_optimizer((2,), (1,), hsdp_replicate_dedup=False)
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    reference = Muon(reference_params)
    with patch("hyper_parallel.core.optimizer.optimizer.dist") as mock_dist:
        for step in range(3):
            for param, ref_param in zip(params, reference_params):
                param.grad = torch.full_like(param, 0.1 * (step + 1))
                ref_param.grad = param.grad.clone()
            optimizer.step()
            reference.step()
            for param, ref_param in zip(params, reference_params):
                torch.testing.assert_close(param, ref_param)
                torch.testing.assert_close(
                    optimizer.state[param]["momentum_buffer"], reference.state[ref_param]["momentum_buffer"],
                )
        mock_dist.broadcast.assert_not_called()


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

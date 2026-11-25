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
"""Triangular Cube GEMM and rounded symmetric stores for Ascend NPUs."""

from functools import lru_cache

import torch
import torch_npu
import triton
import triton.language as tl


_MIRROR_BLOCK = 64


@triton.jit
def _cv_matmul_kernel(
    a, b, output,
    n: tl.constexpr, k: tl.constexpr, batch_count: tl.constexpr,
    stride_ab: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bb: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    cores: tl.constexpr,
):
    """Compute lower triangular tiles and reflect each off-diagonal result on writeback."""
    tiles: tl.constexpr = triton.cdiv(n, block_n)
    strips: tl.constexpr = block_n // block_m
    per_batch: tl.constexpr = tiles * (tiles + 1) // 2
    for task in range(tl.program_id(0), batch_count * per_batch * strips, cores):
        owner = task // strips
        batch = owner // per_batch
        local = owner % per_batch
        # Integer decoding keeps tile ownership off the Vector sqrt/workspace path.
        if tiles == 1:
            row = 0
        else:
            low = 0
            high = tiles
            while low + 1 < high:
                middle = (low + high) // 2
                if middle * (middle + 1) // 2 <= local:
                    low = middle
                else:
                    high = middle
            row = low
        column = local - row * (row + 1) // 2
        rows = row * block_n + task % strips * block_m + tl.arange(0, block_m)
        columns = column * block_n + tl.arange(0, block_n)
        reductions = tl.arange(0, block_k)
        batch_index = batch.to(tl.int64)
        left_base = a + batch_index * stride_ab
        right_base = b + batch_index * stride_bb
        accumulator = tl.zeros((block_m, block_n), tl.float32)
        for start in range(triton.cdiv(k, block_k)):
            ks = start * block_k + reductions
            left = tl.load(
                left_base + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=(rows[:, None] < n) & (ks[None, :] < k), other=0,
            )
            right = tl.load(
                right_base + ks[:, None] * stride_bk + columns[None, :] * stride_bn,
                mask=(ks[:, None] < k) & (columns[None, :] < n), other=0,
            )
            accumulator = tl.dot(left, right, accumulator)
        rounded = accumulator.to(output.dtype.element_ty)
        valid = (rows[:, None] < n) & (columns[None, :] < n)
        destination = output + batch_index * n * n
        tl.store(destination + rows[:, None] * n + columns[None, :], rounded, mask=valid)
        if row != column:
            tl.store(destination + columns[:, None] * n + rows[None, :], tl.trans(rounded),
                     mask=(columns[:, None] < n) & (rows[None, :] < n))


@triton.jit
def _mirror_diagonal_kernel(output, n: tl.constexpr, batch_count: tl.constexpr, cores: tl.constexpr,
                            block: tl.constexpr, macro: tl.constexpr):
    """Repair only diagonal GEMM macrotiles after all their strips have landed."""
    groups: tl.constexpr = triton.cdiv(n, macro)
    width: tl.constexpr = macro // block
    for task in range(tl.program_id(0), batch_count * groups * width * width, cores):
        batch = task // (groups * width * width)
        group = task // (width * width) % groups
        row = task // width % width
        column = task % width
        rows = group * macro + row * block + tl.arange(0, block)
        columns = group * macro + column * block + tl.arange(0, block)
        destination = output + batch.to(tl.int64) * n * n
        if row >= column:
            valid = (rows[:, None] < n) & (columns[None, :] < n)
            values = tl.load(destination + rows[:, None] * n + columns[None, :], mask=valid, other=0)
            if row == column:
                # A select keeps the DMA mask rectangular and gives each diagonal tile one writer.
                symmetric = tl.where(rows[:, None] >= columns[None, :], values, tl.trans(values))
                tl.store(destination + rows[:, None] * n + columns[None, :], symmetric, mask=valid)
            else:
                tl.store(destination + columns[:, None] * n + rows[None, :], tl.trans(values),
                         mask=(columns[:, None] < n) & (rows[None, :] < n))


@lru_cache(maxsize=None)
def _core_counts(device: torch.device) -> tuple[int, int]:
    """Reuse static device properties without retaining tensors or streams."""
    properties = torch_npu.npu.get_device_properties(device)
    return properties.cube_core_num, properties.vector_core_num


def _block_shape(n: int, k: int, batch: int) -> tuple[int, int, int]:
    """Select the Cube tile and reduction sizes for the triangular GEMM."""
    if n <= 64:
        return 64, 64, 64
    if n <= 128:
        return 64, 128, 128
    if batch > 1 and n <= 1024 and k >= 2048:
        return 128, 128, 512
    return 128, 256, 256


@lru_cache(maxsize=128)
def _check_indexing_range(
    shape: tuple[int, int, int], blocks: tuple[int, int, int], core_counts: tuple[int, int],
) -> None:
    """Keep tile-local offsets and loop induction within signed 32-bit range."""
    n, k, batch = shape
    block_m, block_n, block_k = blocks
    cube_cores, vector_cores = core_counts
    cube_tiles = triton.cdiv(n, block_n)
    mirror_tiles = triton.cdiv(n, _MIRROR_BLOCK)
    last_n = cube_tiles * block_n - 1
    last_k = triton.cdiv(k, block_k) * block_k - 1
    last_output = max(last_n, mirror_tiles * _MIRROR_BLOCK - 1)
    cube_tasks = batch * (cube_tiles * (cube_tiles + 1) // 2) * (block_n // block_m)
    mirror_tasks = batch * cube_tiles * (block_n // _MIRROR_BLOCK) ** 2
    # Loads can form masked tail addresses, and the last loop increment can step past the task count.
    largest_index = max(
        last_n * k + last_k, last_k * n + last_n, last_output * (n + 1),
        cube_tasks + cube_cores - 1, mirror_tasks + vector_cores - 1,
    )
    if largest_index >= 2**31:
        raise ValueError("Shape exceeds the symmetric_matmul indexing range within a matrix or task grid")


def launch_symmetric_matmul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *, mirror_diagonal: bool,
) -> None:
    """Launch a validated product on its device's current stream.

    Args:
        a: Dense or transposed left NPU operand.
        b: Dense or transposed right NPU operand.
        out: Contiguous destination with the selected output dtype.
        mirror_diagonal: Copy the lower triangle within diagonal tiles after GEMM.
    """
    n, k = a.shape[-2:]
    batch = 1 if a.ndim == 2 else a.shape[0]
    shape = (n, k, batch)
    with torch_npu.npu.device(a.device):
        blocks = _block_shape(n, k, batch)
        core_counts = _core_counts(a.device)
        _check_indexing_range(shape, blocks, core_counts)
        cores, vector_cores = core_counts
        stream = torch_npu.npu.current_stream(a.device)
        for tensor in (a, b, out):
            tensor.record_stream(stream)
        _cv_matmul_kernel[(cores,)](
            a, b, out, n, k, batch,
            0 if a.ndim == 2 else a.stride(0), a.stride(-2), a.stride(-1),
            0 if b.ndim == 2 else b.stride(0), b.stride(-2), b.stride(-1),
            *blocks, cores, num_stages=2,
        )
        if mirror_diagonal:
            _mirror_diagonal_kernel[(vector_cores,)](out, n, batch, vector_cores, _MIRROR_BLOCK, blocks[1])

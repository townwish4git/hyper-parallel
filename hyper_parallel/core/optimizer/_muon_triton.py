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

"""Optional Triton-Ascend kernels; imported only by explicitly selected paths."""
import triton
import triton.language as tl


@triton.jit
def _round_storage(value, dtype: tl.constexpr):
    if dtype == tl.bfloat16:
        # Explicit rounding prevents the backend from removing a cast round-trip.
        bits = value.to(tl.int32, bitcast=True)
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & -65536
        return bits.to(tl.float32, bitcast=True)
    if dtype == tl.float16:
        bits = value.to(tl.int32, bitcast=True)
        rounded = ((bits + 0xfff + ((bits >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)
        scaled = tl.abs(value) * 16777216.0
        lower = tl.floor(scaled)
        fraction = scaled - lower
        increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
        subnormal = (lower + increment.to(tl.float32)) * 0.000000059604644775390625
        subnormal = tl.where(value < 0, -subnormal, subnormal)
        rounded = tl.where(tl.abs(value) < 0.00006103515625, subnormal, rounded)
        overflow = tl.where(value < 0, -float("inf"), float("inf"))
        return tl.where(tl.abs(rounded) >= 65536.0, overflow, rounded)
    return value


@triton.jit
def _momentum_elementwise(grad, buf, output, count: tl.constexpr,
                     momentum1: tl.constexpr, momentum2: tl.constexpr,
                     nesterov: tl.constexpr, block_id, BLOCK: tl.constexpr):
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    dtype = buf.dtype.element_ty
    gradient = tl.load(grad + offsets, mask, other=0).to(tl.float32)
    previous = tl.load(buf + offsets, mask, other=0).to(tl.float32)
    coefficient1 = tl.full((), momentum1, tl.float32)
    coefficient2 = tl.full((), momentum2, tl.float32)
    if dtype == tl.float16:
        coefficient1 = _round_storage(coefficient1, dtype)
        coefficient2 = _round_storage(coefficient2, dtype)
    product = coefficient1 * previous
    if dtype == tl.float16:
        product = _round_storage(product, dtype)
    direction = _round_storage(gradient + product, dtype)
    # Preserve the intermediate storage rounding of the original mul_/add_.
    next_buf = _round_storage(_round_storage(previous * coefficient2, dtype) + gradient, dtype)
    if nesterov:
        direction = _round_storage(_round_storage(direction * coefficient1, dtype) + gradient, dtype)
    tl.store(buf + offsets, next_buf, mask)
    tl.store(output + offsets, direction, mask)


@triton.jit
def _momentum_kernel(g0, g1, g2, g3, b0, b1, b2, b3, o0, o1, o2, o3,
                     count: tl.constexpr, momentum1: tl.constexpr, momentum2: tl.constexpr,
                     nesterov: tl.constexpr, BLOCK: tl.constexpr):
    blocks = tl.cdiv(count, BLOCK)
    tensor_index = tl.program_id(0) // blocks
    block_id = tl.program_id(0) % blocks
    if tensor_index == 0:
        _momentum_elementwise(g0, b0, o0, count, momentum1, momentum2, nesterov, block_id, BLOCK)
    elif tensor_index == 1:
        _momentum_elementwise(g1, b1, o1, count, momentum1, momentum2, nesterov, block_id, BLOCK)
    elif tensor_index == 2:
        _momentum_elementwise(g2, b2, o2, count, momentum1, momentum2, nesterov, block_id, BLOCK)
    else:
        _momentum_elementwise(g3, b3, o3, count, momentum1, momentum2, nesterov, block_id, BLOCK)


@triton.jit
def _apply_elementwise(param, update, count: tl.constexpr, decay: tl.constexpr,
                       alpha: tl.constexpr, block_id, BLOCK: tl.constexpr):
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    previous = tl.load(param + offsets, mask, other=0).to(tl.float32)
    direction = tl.load(update + offsets, mask, other=0).to(tl.float32)
    # Master parameters are FP32; keep decay and addition as separate roundings.
    result = previous * decay + direction * alpha
    tl.store(param + offsets, result, mask)


@triton.jit
def _apply_kernel(p0, p1, p2, p3, u0, u1, u2, u3, count: tl.constexpr,
                  decay: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    blocks = tl.cdiv(count, BLOCK)
    tensor_index = tl.program_id(0) // blocks
    block_id = tl.program_id(0) % blocks
    if tensor_index == 0:
        _apply_elementwise(p0, u0, count, decay, alpha, block_id, BLOCK)
    elif tensor_index == 1:
        _apply_elementwise(p1, u1, count, decay, alpha, block_id, BLOCK)
    elif tensor_index == 2:
        _apply_elementwise(p2, u2, count, decay, alpha, block_id, BLOCK)
    else:
        _apply_elementwise(p3, u3, count, decay, alpha, block_id, BLOCK)

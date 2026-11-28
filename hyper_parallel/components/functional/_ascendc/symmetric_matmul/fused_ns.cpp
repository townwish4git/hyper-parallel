// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifdef HP_ASCEND_950
#include <acl/acl_rt.h>
#endif
#include "symmetric_matmul/product.h"
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
#include "symmetric_matmul/mirror.h"
#endif

namespace {
using BFloat = bfloat16_t;
constexpr int kTileElements = kVectorTile * kVectorTile;
constexpr int kPhaseFlag = 10;

__aicore__ inline void finish_cube_phase() {
  // Product drains the local mirror or epilogue before other cores read its output.
  CrossCoreSetFlag<0x0, PIPE_FIX>(kPhaseFlag);
  CrossCoreWaitFlag<0x0>(kPhaseFlag);
}

__aicore__ inline int update_task_count(const SymmetricPlan &gram) {
  return gram.batch * ((gram.n + gram.tile - 1) / gram.tile) * ((gram.k + kCubeTile - 1) / kCubeTile);
}

__aicore__ inline TaskRegion update_region(const SymmetricPlan &gram, int task) {
  const int row_tiles = (gram.n + gram.tile - 1) / gram.tile;
  const int col_tiles = (gram.k + kCubeTile - 1) / kCubeTile;
  const int row = (task / col_tiles % row_tiles) * gram.tile;
  const int col = (task % col_tiles) * kCubeTile;
  return {task / col_tiles / row_tiles, row, col, min(gram.tile, gram.n - row), min(kCubeTile, gram.k - col)};
}

__aicore__ inline void update_product(GM_ADDR x, GM_ADDR poly, GM_ADDR out, const FusedNsPlan &plan) {
  TPipe pipe;
  using Operand = matmul::MatmulType<TPosition::GM, CubeFormat::ND, BFloat, false>;
  matmul::MatmulImpl<Operand, Operand, Operand, Operand, CFG_MDL> mm;
  TCubeTiling tiling;
  load_cube_tiling(tiling, plan.update_tiling);
  mm.SetSubBlockIdx(0);
  mm.Init(&tiling, &pipe);

  GlobalTensor<BFloat> input, coefficients, output;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ BFloat *>(x));
  coefficients.SetGlobalBuffer(reinterpret_cast<__gm__ BFloat *>(poly));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ BFloat *>(out));
  const auto &gram = plan.gram;
  const int task_count = update_task_count(gram);
  int iteration = 0;
  for (int task = GetBlockIdx(); task < task_count; task += gram.cores, ++iteration) {
    const int slot = iteration % 2;
    if (iteration >= 2) {
      wait_tile_consumed(slot);
    }
    const auto region = update_region(gram, task);
    const int matrix_base = region.batch * gram.n * gram.k;
    const int square_base = region.batch * gram.n * gram.n;
    mm.SetSingleShape(region.rows, region.columns, gram.n);
    mm.SetTensorA(coefficients[square_base + region.row * gram.n], false);
    mm.SetTensorB(input[matrix_base + region.col], false);
    mm.IterateAll(output[matrix_base + region.row * gram.k + region.col]);
    mm.End();
    signal_tile_ready(slot);
  }
  for (int slot = 0; slot < 2 && slot < iteration; ++slot) {
    wait_tile_consumed(slot);
  }
  pipe.Destroy();
}

class VectorEpilogue {
 public:
  __aicore__ inline explicit VectorEpilogue(TPipe &pipe) {
    pipe.InitBuffer(first_, kTileElements * sizeof(BFloat));
    pipe.InitBuffer(second_, kTileElements * sizeof(BFloat));
    pipe.InitBuffer(first_float_, kTileElements * sizeof(float));
    pipe.InitBuffer(second_float_, kTileElements * sizeof(float));
  }

  template <bool Polynomial>
  __aicore__ inline void run(GM_ADDR src, GM_ADDR dst, const FusedNsPlan &plan) {
    GlobalTensor<BFloat> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ BFloat *>(src));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ BFloat *>(dst));
    const auto &gram = plan.gram;
    const int stride = Polynomial ? gram.n : gram.k;
    const int count = Polynomial ? task_count(plan.square) : update_task_count(gram);
    const auto gap = static_cast<uint16_t>((stride - kVectorTile) * sizeof(BFloat) / 32);
    int iteration = 0;
    for (int task = GetBlockIdx() / 2; task < count; task += gram.cores, ++iteration) {
      auto region = Polynomial ? task_region(plan.square, task) : update_region(gram, task);
      const int slot = iteration % 2;
      wait_tile_ready(slot);
      int first_row = GetSubBlockIdx() * kVectorTile;
      int row_step = 2 * kVectorTile;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
      if (Polynomial && is_off_diagonal(region)) {
        // Fixpipe has published both halves, so each AIV owns one whole tile.
        if (GetSubBlockIdx() != 0) {
          region = {region.batch, region.col, region.row, region.columns, region.rows};
        }
        first_row = 0;
        row_step = kVectorTile;
      }
#endif
      const int base = region.batch * gram.n * stride;
      for (int i = first_row; i < region.rows; i += row_step) {
        const int row = region.row + i;
        for (int j = 0; j < region.columns; j += kVectorTile) {
          const int col = region.col + j;
          const int offset = base + row * stride + col;
          auto first = first_.Get<BFloat>();
          auto second = second_.Get<BFloat>();
          DataCopy(first, input[offset], {kVectorTile, 4, gap, 0});
          DataCopy(second, output[offset], {kVectorTile, 4, gap, 0});
          PipeBarrier<PIPE_ALL>();
          combine<Polynomial>(first, second, plan);
          // Each tile is consumed once and can be replaced in the Cube output buffer.
          DataCopy(output[offset], first, {kVectorTile, 4, 0, gap});
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
          if (Polynomial && is_off_diagonal(region)) {
            transpose_tile(first, second);
            DataCopy(output[base + col * stride + row], second, {kVectorTile, 4, 0, gap});
          }
#endif
          PipeBarrier<PIPE_ALL>();
        }
      }
      signal_tile_consumed(slot);
    }
  }

 private:
  template <bool Polynomial>
  __aicore__ inline void combine(LocalTensor<BFloat> &first, LocalTensor<BFloat> &second, const FusedNsPlan &plan) {
    auto lhs = first_float_.Get<float>();
    auto rhs = second_float_.Get<float>();
    Cast(lhs, first, RoundMode::CAST_NONE, kTileElements);
    Cast(rhs, second, RoundMode::CAST_NONE, kTileElements);
    PipeBarrier<PIPE_V>();
    Muls(lhs, lhs, Polynomial ? plan.b : plan.a, kTileElements);
    if (Polynomial) {
      Muls(rhs, rhs, plan.c, kTileElements);
    }
    PipeBarrier<PIPE_V>();
    // The Torch path rounds each scaled operand before the addition.
    Cast(first, lhs, RoundMode::CAST_RINT, kTileElements);
    if (Polynomial) {
      Cast(second, rhs, RoundMode::CAST_RINT, kTileElements);
    }
    PipeBarrier<PIPE_V>();
    Cast(lhs, first, RoundMode::CAST_NONE, kTileElements);
    if (Polynomial) {
      Cast(rhs, second, RoundMode::CAST_NONE, kTileElements);
    }
    PipeBarrier<PIPE_V>();
    Add(lhs, lhs, rhs, kTileElements);
    PipeBarrier<PIPE_V>();
    Cast(first, lhs, RoundMode::CAST_RINT, kTileElements);
    PipeBarrier<PIPE_ALL>();
  }

  TBuf<TPosition::VECCALC> first_, second_, first_float_, second_float_;
};
}  // namespace

extern "C"
#ifdef HP_ASCEND_950
// The global phase barriers require all Cube blocks to be scheduled together.
__schedmode__(1) __global__ __mix__(1, 2) void hp_fused_ns_step(
#else
__global__ __aicore__ void hp_fused_ns_step(
#endif
    GM_ADDR x, GM_ADDR gram, GM_ADDR poly, GM_ADDR out, FusedNsPlan plan) {
#ifndef HP_ASCEND_950
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
  if ASCEND_IS_AIC {
    product<BFloat, false, true>(x, x, gram, plan.gram);
    finish_cube_phase();
    product<BFloat, false, false, true>(gram, gram, poly, plan.square);
    finish_cube_phase();
    update_product(x, poly, out, plan);
  } else {
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
    mirror<BFloat>(gram, plan.gram);
#endif
    TPipe pipe;
    VectorEpilogue epilogue(pipe);
    epilogue.run<true>(gram, poly, plan);
    epilogue.run<false>(x, out, plan);
    pipe.Destroy();
  }
}

#ifdef HP_ASCEND_950
extern "C" uint32_t aclrtlaunch_hp_fused_ns_step(
    uint32_t blocks, void *stream, void *x, void *gram, void *poly, void *out, FusedNsPlan *plan) {
  hp_fused_ns_step<<<blocks, nullptr, stream>>>(static_cast<uint8_t *>(x), static_cast<uint8_t *>(gram),
                                              static_cast<uint8_t *>(poly), static_cast<uint8_t *>(out), *plan);
  return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
#endif

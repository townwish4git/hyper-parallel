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
#pragma once
#include "lib/matmul_intf.h"
#include "symmetric_matmul/params.h"
#include "symmetric_matmul/schedule.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
template <class T>
__aicore__ inline void dual_store(const __gm__ void *address, const LocalTensor<int8_t> &acc,
                                  const void *params, const uint64_t, const uint64_t batch_base) {
  const auto &copy = *static_cast<const DataCopyOutParams *>(params);
  auto height = static_cast<uint16_t>(copy.burstLen / 2);
  auto quant = IsSameType<T, half>::value ? QuantMode_t::F322F16 : QuantMode_t::F322BF16;
  auto ptr = reinterpret_cast<__gm__ T *>(const_cast<__gm__ void *>(address));
  GlobalTensor<T> dst;
  dst.SetGlobalBuffer(ptr);
  FixpipeParamsC310<CO2Layout::ROW_MAJOR> lower(copy.oriNSize, height, height, copy.dstStride);
  lower.quantPre = quant;
  Fixpipe<T, float, CFG_ROW_MAJOR>(dst, acc.ReinterpretCast<float>(), lower);
  if (batch_base != 0) {
    auto base = reinterpret_cast<__gm__ T *>(batch_base);
    int64_t offset = ptr - base;
    dst.SetGlobalBuffer(base + (offset % copy.dstStride) * copy.dstStride + offset / copy.dstStride);
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> upper(copy.oriNSize, height, height, copy.dstStride);
    upper.params.srcNzC0Stride = 1;
    upper.quantPre = quant;
    Fixpipe<T, float, CFG_COLUMN_MAJOR>(dst, acc.ReinterpretCast<float>(), upper);
  }
}

constexpr auto dual_store_config = []() {
  auto config = CFG_MDL;
  config.enUnitFlag = false;
  return config;
}();
#endif

template <bool TransA, bool TransB, class Matmul, class T, class Output>
__aicore__ inline void produce_region(Matmul &mm, GlobalTensor<T> &a, GlobalTensor<T> &b, GlobalTensor<Output> &out,
                                      const SymmetricPlan &plan, const TaskRegion &region) {
  if (region.rows == 0) return;
  int64_t input_batch = static_cast<int64_t>(region.batch) * plan.n * plan.k;
  int64_t output_batch = static_cast<int64_t>(region.batch) * plan.n * plan.n;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  // Matmul can subdivide a macro tile. The batch base lets Fixpipe transpose
  // writes from any sub-tile of an off-diagonal macro tile.
  mm.SetSelfDefineData(is_off_diagonal(region) ? reinterpret_cast<uint64_t>(out[output_batch].GetPhyAddr()) : 0UL);
#endif
  mm.SetSingleShape(region.rows, region.columns, plan.k);
  mm.SetTensorA(a[input_batch + (TransA ? region.row : region.row * plan.k)], TransA);
  mm.SetTensorB(b[input_batch + (TransB ? region.col * plan.k : region.col)], TransB);
  mm.IterateAll(out[output_batch + region.row * plan.n + region.col]);
  mm.End();
}

__aicore__ inline void load_cube_tiling(TCubeTiling &tiling, const uint32_t *data) {
  auto *words = reinterpret_cast<uint32_t *>(&tiling);
  for (uint32_t i = 0; i < sizeof(tiling) / sizeof(uint32_t); ++i) {
    words[i] = data[i];
  }
}

template <class T, bool TransA, bool TransB, bool NotifyAllTiles = false>
__aicore__ inline void product(GM_ADDR a, GM_ADDR b, GM_ADDR out, SymmetricPlan &plan) {
  if ASCEND_IS_AIV {
    return;
  }
  TPipe pipe;
  using Left = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, TransA>;         // codespell:ignore nd
  using Right = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, TransB>;  // codespell:ignore nd
  using Result = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;  // codespell:ignore nd
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  matmul::MatmulImpl<Left, Right, Result, Result, dual_store_config, MatmulCallBackFunc<dual_store<T>>> mm;
#else
  matmul::MatmulImpl<Left, Right, Result, Result, CFG_MDL> mm;
#endif
  TCubeTiling tiling;
  load_cube_tiling(tiling, plan.tiling);
  mm.SetSubBlockIdx(0);
  mm.Init(&tiling, &pipe);
  GlobalTensor<T> ga, gb;
  GlobalTensor<T> gc;
  ga.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));
  gb.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b));
  gc.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
  int tasks = task_count(plan);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  constexpr bool vector_mirror = false;
#else
  constexpr bool vector_mirror = true;
#endif
  // Both AIVs must acknowledge a tile before the flag slot can be reused.
  int iteration = 0;
  for (int task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
    auto region = task_region(plan, task);
    const bool notify = NotifyAllTiles || (vector_mirror && is_off_diagonal(region));
    if (notify && iteration >= 2) {
      wait_tile_consumed(iteration % 2);
    }
    produce_region<TransA, TransB>(mm, ga, gb, gc, plan, region);
    if (notify) {
      signal_tile_ready(iteration % 2);
      ++iteration;
    }
  }
  for (int slot = 0; slot < 2 && slot < iteration; ++slot) {
    wait_tile_consumed(slot);
  }
  pipe.Destroy();
}

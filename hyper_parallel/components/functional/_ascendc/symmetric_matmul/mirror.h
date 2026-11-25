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
#include "symmetric_matmul/params.h"
#include "symmetric_matmul/schedule.h"

__aicore__ inline void initialize_diagonal_mask(TBuf<TPosition::VECCALC> &selection) {
  constexpr int tile = 64;
  auto bits = selection.Get<uint64_t>();
  for (int row = 0; row < tile; ++row) {
    bits.SetValue(row, row == 63 ? ~static_cast<uint64_t>(0) : (static_cast<uint64_t>(1) << (row + 1)) - 1);
  }
}

template <class T>
__aicore__ inline void transpose_tile(LocalTensor<T> &r, LocalTensor<T> &tr) {
  constexpr int tile = 64;
  uint64_t sources[16], destinations[16];
  for (int block = 0; block < tile / 16; ++block) {
    for (int line = 0; line < 16; ++line) {
      sources[line] = reinterpret_cast<uint64_t>(r[(block * 16 + line) * tile].GetPhyAddr());
      destinations[line] = reinterpret_cast<uint64_t>(tr[line * tile + block * 16].GetPhyAddr());
    }
    TransDataTo5HD<uint16_t>(destinations, sources, {false, false, tile / 16, tile, 1});
  }
  PipeBarrier<PIPE_V>();
}


template <class T>
__aicore__ inline void mirror(GM_ADDR out, const SymmetricPlan &plan) {
  TPipe pipe;
  constexpr int tile = 64;
  TBuf<TPosition::VECCALC> rounded, transposed, selection;
  pipe.InitBuffer(rounded, tile * tile * sizeof(T));
  pipe.InitBuffer(transposed, tile * tile * sizeof(T));
  pipe.InitBuffer(selection, tile * tile / 8);
  initialize_diagonal_mask(selection);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  TBuf<TPosition::VECCALC> offsets;
  pipe.InitBuffer(offsets, tile * tile * sizeof(uint32_t));
  auto indices = offsets.Get<int32_t>();
  for (int col = 0; col < tile; ++col) indices.SetValue(col, col * tile * sizeof(T));
  SetFlag<HardEvent::S_V>(EVENT_ID0);
  WaitFlag<HardEvent::S_V>(EVENT_ID0);
  for (int row = 1; row < tile; ++row) {
    Adds(indices[row * tile], indices, static_cast<int32_t>(row * sizeof(T)), tile);
  }
  PipeBarrier<PIPE_V>();
#endif
  auto lower = rounded.Get<T>();
  auto upper = transposed.Get<T>();
  GlobalTensor<T> dst;
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
  auto gap = static_cast<uint16_t>((plan.n - tile) * sizeof(T) / 32);
  int iteration = 0;
  for (int task = GetBlockIdx() / 2; task < task_count(plan); task += plan.cores, ++iteration) {
    auto region = task_region(plan, task);
    CrossCoreWaitFlag<0x2>(6 + iteration % 2);
    int64_t base = static_cast<int64_t>(region.batch) * plan.n * plan.n;
    for (int i = GetSubBlockIdx() * tile; i < region.rows; i += 2 * tile) {
      int row = region.row + i;
      for (int j = 0; j < region.columns && region.col + j <= row; j += tile) {
        int col = region.col + j;
        DataCopy(lower, dst[base + row * plan.n + col], {tile, 4, gap, 0});
        PipeBarrier<PIPE_ALL>();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(upper.template ReinterpretCast<uint16_t>(), lower.template ReinterpretCast<uint16_t>(),
               indices.template ReinterpretCast<uint32_t>(), 0, tile * tile);
        PipeBarrier<PIPE_V>();
#else
        transpose_tile(lower, upper);
#endif
        if (row == col) {
          Select(upper.template ReinterpretCast<half>(), selection.Get<uint8_t>(),
                 lower.template ReinterpretCast<half>(), upper.template ReinterpretCast<half>(),
                 SELMODE::VSEL_TENSOR_TENSOR_MODE, tile * tile);
          PipeBarrier<PIPE_ALL>();
        }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
        DataCopy(dst[base + col * plan.n + row], upper, {tile, 4, 0, gap});
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
#endif
        PipeBarrier<PIPE_ALL>();
      }
    }
    CrossCoreSetFlag<0x2, PIPE_MTE3>(8 + iteration % 2);
  }
}

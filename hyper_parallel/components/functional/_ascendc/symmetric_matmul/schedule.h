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
#include <kernel_operator.h>
#include <algorithm>
#include "symmetric_matmul/params.h"
using namespace AscendC;  // NOLINT(build/namespaces) CANN device intrinsics and templates.

constexpr int kTileReadyFlag = 6;
constexpr int kTileConsumedFlag = 8;

__aicore__ inline void signal_tile_ready(int slot) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  // Mode 4 has separate flag banks for the two AIV consumers.
  CrossCoreSetFlag<0x4, PIPE_FIX>(kTileReadyFlag + slot);
  CrossCoreSetFlag<0x4, PIPE_FIX>(kTileReadyFlag + slot + 16);
#else
  CrossCoreSetFlag<0x2, PIPE_FIX>(kTileReadyFlag + slot);
#endif
}

__aicore__ inline void wait_tile_ready(int slot) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  CrossCoreWaitFlag<0x4>(kTileReadyFlag + slot);
#else
  CrossCoreWaitFlag<0x2>(kTileReadyFlag + slot);
#endif
}

__aicore__ inline void signal_tile_consumed(int slot) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  CrossCoreSetFlag<0x4, PIPE_MTE3>(kTileConsumedFlag + slot);
#else
  CrossCoreSetFlag<0x2, PIPE_MTE3>(kTileConsumedFlag + slot);
#endif
}

__aicore__ inline void wait_tile_consumed(int slot) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  CrossCoreWaitFlag<0x4>(kTileConsumedFlag + slot);
  CrossCoreWaitFlag<0x4>(kTileConsumedFlag + slot + 16);
#else
  CrossCoreWaitFlag<0x2>(kTileConsumedFlag + slot);
#endif
}

__aicore__ inline void decode(int task, int tiles, int &batch, int &row, int &col) {
  int count = tiles * (tiles + 1) / 2;
  batch = task / count;
  int local = task % count;
  int left = 0, right = tiles;
  while (left + 1 < right) {
    int middle = (left + right) / 2;
    if (static_cast<int64_t>(middle) * (middle + 1) / 2 <= local)
      left = middle;
    else
      right = middle;
  }
  row = left;
  col = local - row * (row + 1) / 2;
}

struct TaskRegion {
  int batch, row, col, rows, columns;
};

__aicore__ inline bool is_off_diagonal(const TaskRegion &region) {
  return region.row != region.col;
}

__aicore__ inline int task_count(const SymmetricPlan &plan) {
  int tiles = (plan.n + plan.tile - 1) / plan.tile;
  return plan.batch * tiles * (tiles + 1) / 2;
}

__aicore__ inline TaskRegion task_region(const SymmetricPlan &plan, int task) {
  int batch, row, col;
  decode(task, (plan.n + plan.tile - 1) / plan.tile, batch, row, col);
  row *= plan.tile;
  col *= plan.tile;
  return {batch, row, col, min(plan.tile, plan.n - row), min(plan.tile, plan.n - col)};
}

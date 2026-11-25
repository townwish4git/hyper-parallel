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
#include <algorithm>
#include <cstring>
#include "tiling/matmul/matmul_tiling.h"
#include "symmetric_matmul/params.h"

namespace {
using CubeTiling = AscendC::tiling::TCubeTiling;

bool make_tiling(int n, int k, int rows, int columns, int dtype, bool trans_a, bool trans_b, CubeTiling &result) {
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  matmul_tiling::MatmulApiTiling tiler(*platform);
  using matmul_tiling::CubeFormat;
  using matmul_tiling::DataType;
  using matmul_tiling::TPosition;
  auto type = dtype == 0 ? DataType::DT_FLOAT16 : DataType::DT_BF16;
  tiler.SetAType(TPosition::GM, CubeFormat::ND, type, trans_a);                         // codespell:ignore nd
  tiler.SetBType(TPosition::GM, CubeFormat::ND, type, trans_b);                          // codespell:ignore nd
  tiler.SetCType(TPosition::GM, CubeFormat::ND, type);  // codespell:ignore nd
  tiler.SetShape(rows, columns, k);
  tiler.SetOrgShape(n, n, k);
  tiler.SetMatmulConfigParams(1);
  tiler.SetBias(false);
  tiler.SetBufferSpace(-1, -1, -1);
  return tiler.GetTiling(result) == 0;
}

}  // namespace

SymmetricPlan make_plan(int n, int k, int batch, int dtype, bool trans_a, bool trans_b) {
  SymmetricPlan plan{};
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform == nullptr) return plan;
  int tile = std::min(n, 256);
  CubeTiling tiling;
  if (!make_tiling(n, k, tile, tile, dtype, trans_a, trans_b, tiling)) return plan;
  plan.n = n;
  plan.k = k;
  plan.batch = batch;
  plan.tile = tile;
  plan.dtype = dtype;
  plan.trans_a = trans_a;
  plan.trans_b = trans_b;
  int tiles = (n + tile - 1) / tile;
  plan.cores = std::min<int64_t>(platform->GetCoreNumAic(), static_cast<int64_t>(batch) * tiles * (tiles + 1) / 2);
  static_assert(sizeof(tiling) <= sizeof(plan.tiling));
  std::memcpy(plan.tiling, &tiling, sizeof(tiling));
  return plan;
}

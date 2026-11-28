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
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "symmetric_matmul/params.h"

extern "C" uint32_t aclrtlaunch_hp_symm_pipeline(uint32_t, void *, void *, void *, void *, SymmetricPlan *);
#ifdef HP_ENABLE_FUSED_NS
extern "C" uint32_t aclrtlaunch_hp_fused_ns_step(uint32_t, void *, void *, void *, void *, void *, FusedNsPlan *);
#endif

namespace {
bool plain_format(const at::Tensor &x) {
  auto format = at_npu::native::get_npu_format(x);
  return format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW;
}

bool aligned(const at::Tensor &x) {
  return reinterpret_cast<uintptr_t>(x.data_ptr()) % 32 == 0;
}

at::Tensor pack(const at::Tensor &x, int rows, int cols) {
  if (x.size(-2) == rows && x.size(-1) == cols && aligned(x) &&
      (x.is_contiguous() || x.transpose(-2, -1).is_contiguous())) return x;
  auto shape = x.sizes().vec();
  shape[shape.size() - 2] = rows;
  shape.back() = cols;
  auto packed = at::zeros(shape, x.options());
  packed.narrow(-2, 0, x.size(-2)).narrow(-1, 0, x.size(-1)).copy_(x);
  return packed;
}

void check_operands(const at::Tensor &a, const at::Tensor &b, const at::Tensor &out) {
  TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && a.dim() == b.dim() && out.dim() == a.dim(), "expected 2D/3D matrices");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device(), "mixed devices");
  TORCH_CHECK((a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16) &&
              a.scalar_type() == b.scalar_type() && a.scalar_type() == out.scalar_type(),
              "expected same FP16/BF16 dtype");
  TORCH_CHECK(!at::GradMode::is_enabled() || !(a.requires_grad() || b.requires_grad() || out.requires_grad()),
              "symmetric GEMM is forward-only");
  TORCH_CHECK(!out.is_alias_of(a) && !out.is_alias_of(b), "output must not alias operands");
}

void check_shapes(const at::Tensor &a, const at::Tensor &b, const at::Tensor &out) {
  const auto n = a.size(-2);
  TORCH_CHECK(b.size(-2) == a.size(-1) && b.size(-1) == n && out.size(-2) == n && out.size(-1) == n,
              "invalid matrix dimensions");
  TORCH_CHECK(a.dim() == 2 || (a.size(0) == b.size(0) && a.size(0) == out.size(0)), "invalid batch dimensions");
  TORCH_CHECK(out.is_contiguous(), "expected a contiguous output");
}

at::Tensor symm_out(const at::Tensor &a, const at::Tensor &b, at::Tensor out) {
  check_operands(a, b, out);
  check_shapes(a, b, out);
  c10::DeviceGuard guard(a.device());
  TORCH_CHECK(plain_format(a) && plain_format(b) && plain_format(out), "expected plain NPU storage format");
  if (!out.numel()) return out;
  const int64_t n = a.size(-2), k = a.size(-1), batch = a.dim() == 2 ? 1 : a.size(0);
  if (!k) return out.zero_();
  TORCH_CHECK(n <= 46340 && k <= INT32_MAX - 16 && batch <= INT32_MAX, "unsupported matrix size");
  const int pn = (n + 63) / 64 * 64, pk = (k + 15) / 16 * 16;
  TORCH_CHECK(int64_t(pn) * std::max(pn, pk) <= INT32_MAX && batch * ((pn + 63) / 64) * ((pn + 63) / 64 + 1)
              <= INT32_MAX, "matrix indices exceed int32");
  auto left = pack(a, pn, pk), right = pack(b, pk, pn);
  auto shape = out.sizes().vec();
  shape[shape.size() - 2] = shape.back() = pn;
  auto dst = pn == n && aligned(out) ? out : at::empty(shape, out.options());
  bool ta = !left.is_contiguous(), tb = !right.is_contiguous();
  int dtype = a.scalar_type() == at::kBFloat16;
  const std::array<int, 7> key{a.device().index(), pn, pk, static_cast<int>(batch), dtype, ta, tb};
  // Each submission thread retains its own cache. Queue closures own a copy of their plan.
  static thread_local std::map<std::array<int, 7>, SymmetricPlan> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    auto plan = make_plan(pn, pk, batch, dtype, ta, tb);
    TORCH_CHECK(plan.cores > 0, "Ascend C tiling failed");
    if (cache.size() >= 128) cache.clear();
    it = cache.emplace(key, plan).first;
  }
  auto stream = c10_npu::getCurrentNPUStream(a.device().index());
  auto raw_stream = stream.stream(false);
  auto submit = [plan = it->second, raw_stream, left, right, dst]() mutable -> int {
    return aclrtlaunch_hp_symm_pipeline(plan.cores, raw_stream, left.data_ptr(), right.data_ptr(),
                                      dst.data_ptr(), &plan);
  };
  at_npu::native::OpCommand::RunOpApiV3("hp_symm_pipeline", submit, false, &stream);
  if (!dst.is_same(out)) out.copy_(dst.narrow(-2, 0, n).narrow(-1, 0, n));
  return out;
}

#ifdef HP_ENABLE_FUSED_NS
void check_ns_buffers(const at::Tensor &x, const at::Tensor &gram, const at::Tensor &poly, const at::Tensor &out) {
  const std::array<const at::Tensor *, 4> tensors{&x, &gram, &poly, &out};
  for (size_t index = 0; index < tensors.size(); ++index) {
    const auto &tensor = *tensors[index];
    TORCH_CHECK(tensor.device() == x.device() && tensor.scalar_type() == at::kBFloat16,
                "fused Newton-Schulz expects all tensors on the same NPU with BF16 dtype");
    TORCH_CHECK(tensor.is_contiguous() && aligned(tensor) && plain_format(tensor),
                "fused Newton-Schulz expects 32-byte-aligned contiguous tensors with plain NPU storage");
    TORCH_CHECK(!at::GradMode::is_enabled() || !tensor.requires_grad(), "fused Newton-Schulz is forward-only");
    for (size_t previous = 0; previous < index; ++previous) {
      TORCH_CHECK(!tensor.is_alias_of(*tensors[previous]), "fused Newton-Schulz buffers must have distinct storage");
    }
  }
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "fused Newton-Schulz expects a matrix or a matrix batch");
  auto square_shape = x.sizes().vec();
  square_shape.back() = x.size(-2);
  const c10::IntArrayRef square_sizes(square_shape);
  TORCH_CHECK(gram.sizes() == square_sizes && poly.sizes() == square_sizes && out.sizes() == x.sizes(),
              "fused Newton-Schulz buffer shapes do not match the input");
}

at::Tensor fused_ns_step_out(const at::Tensor &x, double a, double b, double c,
                             at::Tensor gram, at::Tensor poly, at::Tensor out) {
  c10::DeviceGuard guard(x.device());
  check_ns_buffers(x, gram, poly, out);
  const int64_t n = x.size(-2), k = x.size(-1);
  constexpr int64_t max_stride = std::numeric_limits<uint16_t>::max() * 16L + kVectorTile;
  TORCH_CHECK(x.numel() > 0 && x.numel() <= std::numeric_limits<int32_t>::max() &&
              n <= k && k <= max_stride && n % kVectorTile == 0 && k % kVectorTile == 0,
              "fused Newton-Schulz requires nonempty matrices with 64-aligned dimensions, n <= k and 32-bit indexing");
  const int batch = x.dim() == 2 ? 1 : static_cast<int>(x.size(0));
  const std::array<int, 4> key{x.device().index(), static_cast<int>(n), static_cast<int>(k), batch};
  static thread_local std::map<std::array<int, 4>, FusedNsPlan> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    auto plan = make_fused_ns_plan(n, k, batch);
    TORCH_CHECK(plan.gram.cores > 0, "Ascend C fused Newton-Schulz tiling failed");
    if (cache.size() >= 128) {
      cache.clear();
    }
    it = cache.emplace(key, plan).first;
  }
  auto plan = it->second;
  plan.a = static_cast<float>(a);
  plan.b = static_cast<float>(b);
  plan.c = static_cast<float>(c);
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto raw_stream = stream.stream(false);
  auto submit = [plan, raw_stream, x, gram, poly, out]() mutable -> int {
    return aclrtlaunch_hp_fused_ns_step(plan.gram.cores, raw_stream, x.data_ptr(), gram.data_ptr(),
                                      poly.data_ptr(), out.data_ptr(), &plan);
  };
  at_npu::native::OpCommand::RunOpApiV3("hp_fused_ns_step", submit, false, &stream);
  return out;
}
#endif
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("symmetric_gemm_mirror.out(Tensor a, Tensor b, *, Tensor(a!) out) -> Tensor(a!)");
#ifdef HP_ENABLE_FUSED_NS
  m.def("fused_ns_step.out(Tensor x, float a, float b, float c, *, "
        "Tensor(a!) gram, Tensor(b!) poly, Tensor(c!) out) -> Tensor(c!)");
#endif
}
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("symmetric_gemm_mirror.out", TORCH_FN(symm_out));
#ifdef HP_ENABLE_FUSED_NS
  m.impl("fused_ns_step.out", TORCH_FN(fused_ns_step_out));
#endif
}

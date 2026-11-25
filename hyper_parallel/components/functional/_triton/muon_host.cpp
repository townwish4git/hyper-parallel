// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace py = pybind11;

// The companion launcher uses Triton-Ascend's argument packing, profiling and cross-core lock handling.
using Launch = void (*)(const char*, void*, void*, int, int, int, const int64_t*, const int*, int, const int*,
                        const void* const*, const size_t*, int, void*, uint64_t, int*);

class ProductPlan {
 public:
  ProductPlan(uintptr_t launch, uintptr_t function, const std::string& name, int grid, int64_t workspace,
              const std::array<int, 3>& kinds, const at::Tensor& lhs, const at::Tensor& rhs, const at::Tensor& out)
      : launch_(reinterpret_cast<Launch>(launch)), function_(reinterpret_cast<void*>(function)), name_(name),
        grid_(grid), workspace_(workspace), kinds_(kinds), device_(lhs.device()), shape_(lhs.sizes().vec()),
        strides_(lhs.strides().vec()), output_(out.sizes().vec()) {
    TORCH_CHECK(launch_ && function_ && grid_ > 0 && workspace_ >= 0, "Invalid Triton Muon launcher");
    size_t index = 0;
    for (const auto* tensor : {&lhs, &rhs, &out}) {
      dims_[index++] = static_cast<int>(tensor->dim());
      shapes_.insert(shapes_.end(), tensor->sizes().begin(), tensor->sizes().end());
    }
  }

  int64_t workspace() const { return workspace_; }

  void check(const at::Tensor& input, const at::Tensor& out) const {
    TORCH_CHECK(input.device() == device_ && out.device() == device_ && input.scalar_type() == at::kBFloat16 &&
                out.scalar_type() == at::kBFloat16 && input.sizes() == shape_ && input.strides() == strides_ &&
                out.sizes() == output_ && out.is_contiguous() && !input.is_alias_of(out) &&
                reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "Incompatible Triton Muon buffers");
  }

  int run(void* input, void* out, void* stream, const at::Tensor& scratch) const {
    // The compiled strides distinguish X @ X.T from A @ A. Both operands start at the same address.
    const void* args[] = {&input, &input, &out};
    constexpr size_t sizes[] = {sizeof(void*), sizeof(void*), sizeof(void*)};
    int status = -1;
    launch_(name_.c_str(), function_, stream, grid_, 1, 1, shapes_.data(), dims_.data(), 3, kinds_.data(),
            args, sizes, 3, scratch.data_ptr(), scratch.numel(), &status);
    return status;
  }

 private:
  Launch launch_;
  void* function_;
  std::string name_;
  int grid_;
  int64_t workspace_;
  std::array<int, 3> kinds_, dims_;
  c10::Device device_;
  std::vector<int64_t> shape_, strides_, output_, shapes_;
};

void iterate(std::shared_ptr<ProductPlan> gram, std::shared_ptr<ProductPlan> square, at::Tensor x, at::Tensor a,
             at::Tensor b, at::Tensor c, const std::vector<std::array<double, 3>>& coefficients) {
  TORCH_CHECK(!at::GradMode::is_enabled(), "Paired Muon dispatch requires no_grad");
  gram->check(x, a);
  square->check(a, b);
  TORCH_CHECK(c.device() == x.device() && c.scalar_type() == x.scalar_type() && c.sizes() == x.sizes() &&
              c.is_contiguous() && !c.is_alias_of(x) && !c.is_alias_of(a) && !c.is_alias_of(b) && !b.is_alias_of(x),
              "Invalid Muon update buffer");
  c10::DeviceGuard guard(x.device());
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto raw_stream = stream.stream(false);
  // The caller allocates all four buffers on this stream. Queue callbacks own their operands until submission.
  auto scratch = at::empty({std::max(gram->workspace(), square->workspace())}, x.options().dtype(at::kByte));
  for (const auto& coeff : coefficients) {
    auto submit_pair = [gram, square, x, a, b, raw_stream, scratch]() -> int {
      int status = gram->run(x.data_ptr(), a.data_ptr(), raw_stream, scratch);
      return status == 0 ? square->run(a.data_ptr(), b.data_ptr(), raw_stream, scratch) : status;
    };
    at_npu::native::OpCommand::RunOpApiV3("hp_muon_symm_pair", submit_pair, false, &stream);
    a.mul_(coeff[1]);
    b.mul_(coeff[2]);
    a.add_(b);
    at::matmul_out(c, a, x);
    x.mul_(coeff[0]).add_(c);
  }
}
}  // namespace

PYBIND11_MODULE(_hp_muon_host, module) {
  py::class_<ProductPlan, std::shared_ptr<ProductPlan>>(module, "ProductPlan", py::module_local())
      .def(py::init<uintptr_t, uintptr_t, const std::string&, int, int64_t, const std::array<int, 3>&,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&>());
  module.def("iterate", &iterate);
}

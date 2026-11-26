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
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define ASCENDC_CUBE_ONLY
#endif
#include "symmetric_matmul/product.h"
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
#include "symmetric_matmul/mirror.h"
#endif

template <class T>
__aicore__ inline void dispatch(GM_ADDR a, GM_ADDR b, GM_ADDR out, SymmetricPlan &plan) {
  if ASCEND_IS_AIC {
    if (plan.trans_a) {
      if (plan.trans_b) product<T, true, true>(a, b, out, plan);
      else product<T, true, false>(a, b, out, plan);
    } else {
      if (plan.trans_b) product<T, false, true>(a, b, out, plan);
      else product<T, false, false>(a, b, out, plan);
    }
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
  } else {
    mirror<T>(out, plan);
#endif
  }
}

extern "C" __global__ __aicore__ void hp_symm_pipeline(GM_ADDR a, GM_ADDR b, GM_ADDR out, SymmetricPlan plan) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#else
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
  if (plan.dtype == 0) dispatch<half>(a, b, out, plan);
  else dispatch<bfloat16_t>(a, b, out, plan);
}

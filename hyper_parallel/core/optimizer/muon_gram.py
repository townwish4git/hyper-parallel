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

"""Experimental Gram-space Newton-Schulz with bounded temporary storage."""
import torch  # pylint: disable=forbidden-backend-import


def _gram_newton_schulz(normalized, coefficients, reset_iterations, compute_dtype):
    """Iterate on the smaller Gram matrix without renormalizing at restarts."""
    original_shape = normalized.shape
    original_dtype = normalized.dtype
    dtype = torch.float64 if original_dtype == torch.float64 else getattr(torch, compute_dtype)
    mat_x = normalized.reshape(-1, *normalized.shape[-2:]).to(dtype)
    if not coefficients:
        return normalized.clone()
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 1 for index in reset_iterations):
        raise ValueError("Gram restart indices must be positive integers")
    resets = set(reset_iterations)
    gram = torch.bmm(mat_x, mat_x.mT)
    polynomial = torch.empty_like(gram)
    transform = torch.empty_like(gram)
    transform_next = torch.empty_like(gram)
    gram_product = torch.empty_like(gram)
    gram_next = torch.empty_like(gram)
    next_x = torch.empty_like(mat_x)
    for index, (coeff_a, coeff_b, coeff_c) in enumerate(coefficients):
        if index in resets:
            torch.bmm(transform, mat_x, out=next_x)
            mat_x, next_x = next_x, mat_x
            torch.bmm(mat_x, mat_x.mT, out=gram)
        torch.baddbmm(gram, gram, gram, beta=coeff_b, alpha=coeff_c, out=polynomial)
        if index == 0 or index in resets:
            transform.copy_(polynomial)
            transform.diagonal(dim1=-2, dim2=-1).add_(coeff_a)
        else:
            torch.baddbmm(transform, transform, polynomial, beta=coeff_a, out=transform_next)
            transform, transform_next = transform_next, transform
        if index + 1 < len(coefficients) and index + 1 not in resets:
            torch.baddbmm(gram, gram, polynomial, beta=coeff_a, out=gram_product)
            torch.baddbmm(gram_product, polynomial, gram_product, beta=coeff_a, out=gram_next)
            gram, gram_next = gram_next, gram
    torch.bmm(transform, mat_x, out=next_x)
    return next_x.to(original_dtype).reshape(original_shape)

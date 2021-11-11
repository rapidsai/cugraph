/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cugraph/prims/property_op_utils.cuh>

namespace cugraph {
namespace reduce_op {

// in case there is no payload to reduce
struct null {
  using type = void;
};

// reducing N elements, any element can be a valid output.
template <typename T>
struct any {
  using type = T;
  // FIXME: actually every reduction operation should be side-effect free if reduction is performed
  // by thrust; thrust reduction call rounds up the number of invocations based on the block size
  // and discards the values outside the valid range; this does not work if the reduction operation
  // has side-effects.
  static constexpr bool pure_function = true;  // this can be called in any process

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return lhs; }
};

// FIXME: thrust::minimum can replace this.
// reducing N elements (operator < should be defined between any two elements), the minimum element
// should be selected.
template <typename T>
struct min {
  using type = T;
  // FIXME: actually every reduction operation should be side-effect free if reduction is performed
  // by thrust; thrust reduction call rounds up the number of invocations based on the block size
  // and discards the values outside the valid range; this does not work if the reduction operation
  // has side-effects.
  static constexpr bool pure_function = true;  // this can be called in any process

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
};

// FIXME: thrust::plus can replace this.
// reducing N elements (operator < should be defined between any two elements), the minimum element
// should be selected.
template <typename T>
struct plus {
  using type = T;
  // FIXME: actually every reduction operation should be side-effect free if reduction is performed
  // by thrust; thrust reduction call rounds up the number of invocations based on the block size
  // and discards the values outside the valid range; this does not work if the reduction operation
  // has side-effects.
  static constexpr bool pure_function = true;  // this can be called in any process
  property_op<T, thrust::plus> op{};

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return op(lhs, rhs); }
};

}  // namespace reduce_op
}  // namespace cugraph

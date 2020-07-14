/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cugraph {
namespace experimental {
namespace detail {
namespace reduce_op {

// reducing N elements, any element can be a valid output.
template <typename T>
struct any {
  using type = T;
  static constexpr bool pure_function = true;  // this can be called in any process

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return lhs; }
};

// reducing N elements (operator < should be defined between any two elements), the minimum element should be selected.
template <typename T>
struct min {
  using type = T;
  static constexpr bool pure_function = true;  // this can be called in any process

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
};

}  // namespace reduce_op
}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

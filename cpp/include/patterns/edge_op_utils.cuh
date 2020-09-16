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

#include <utilities/thrust_tuple_utils.cuh>

#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <array>
#include <type_traits>

namespace cugraph {
namespace experimental {

template <typename ResultOfEdgeOp, typename Enable = void>
struct is_valid_edge_op {
  static constexpr bool value = false;
};

template <typename ResultOfEdgeOp>
struct is_valid_edge_op<
  ResultOfEdgeOp,
  typename std::conditional<false, typename ResultOfEdgeOp::type, void>::type> {
  static constexpr bool valid = true;
};

template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp>
struct evaluate_edge_op {
  using vertex_type    = typename GraphViewType::vertex_type;
  using weight_type    = typename GraphViewType::weight_type;
  using row_value_type = typename std::iterator_traits<AdjMatrixRowValueInputIterator>::value_type;
  using col_value_type = typename std::iterator_traits<AdjMatrixColValueInputIterator>::value_type;

  template <typename V = vertex_type,
            typename W = weight_type,
            typename R = row_value_type,
            typename C = col_value_type,
            typename E = EdgeOp>
  __device__ std::enable_if_t<is_valid_edge_op<typename std::result_of<E(V, V, W, R, C)>>::valid,
                              typename std::result_of<E(V, V, W, R, C)>::type>
  compute(V r, V c, W w, R rv, C cv, E e)
  {
    return e(r, c, w, rv, cv);
  }

  template <typename V = vertex_type,
            typename W = weight_type,
            typename R = row_value_type,
            typename C = col_value_type,
            typename E = EdgeOp>
  __device__ std::enable_if_t<is_valid_edge_op<typename std::result_of<E(V, V, R, C)>>::valid,
                              typename std::result_of<E(V, V, R, C)>::type>
  compute(V r, V c, W w, R rv, C cv, E e)
  {
    return e(r, c, rv, cv);
  }
};

template <typename T>
__host__ __device__ std::enable_if_t<std::is_arithmetic<T>::value, T> plus_edge_op_result(
  T const& lhs, T const& rhs)
{
  return lhs + rhs;
}

template <typename T>
__host__ __device__ std::enable_if_t<is_thrust_tuple<T>::value, T> plus_edge_op_result(T const& lhs,
                                                                                       T const& rhs)
{
  return plus_thrust_tuple<T>()(lhs, rhs);
}

}  // namespace experimental
}  // namespace cugraph

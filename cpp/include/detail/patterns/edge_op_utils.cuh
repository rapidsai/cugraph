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

#include <cub/cub.cuh>
#include <thrust/tuple.h>
 
#include <array>
#include <type_traits>
 
namespace { 
}  // namespace
 
namespace cugraph {
namespace experimental {
namespace detail {

template <typename ResultOfEdgeOp, typename Enable = void> 
struct is_valid_edge_op {
  static constexpr bool value = false;
};
    
template <typename ResultOfEdgeOp>
struct is_valid_edge_op<
  ResultOfEdgeOp, typename std::conditional<false, typename ResultOfEdgeOp::type, void>::type
> {
  static constexpr bool valid = true;
};
    
template <typename GraphType,
          typename EdgeOp,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename Enable = void>
struct evaluate_edge_op {
};

template <typename GraphType,
          typename EdgeOp,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator>
struct evaluate_edge_op<
  GraphType, EdgeOp, AdjMatrixRowValueInputIterator, AdjMatrixColValueInputIterator,
  typename std::enable_if_t<GraphType::is_row_major || GraphType::is_column_major>
> {
  using row_value_type =
    typename std::iterator_traits<AdjMatrixRowValueInputIterator>::value_type;
  using col_value_type =
    typename std::iterator_traits<AdjMatrixColValueInputIterator>::value_type;
  using weight_type = typename GraphType::weight_type;

  template <typename E = EdgeOp, typename R = row_value_type, typename C = col_value_type,
            typename W = weight_type>
  __device__
  std::enable_if_t<
    is_valid_edge_op<typename std::result_of<E(R, C, W)>>::valid,
    typename std::result_of<E(R, C, W)>::type
  >
  compute(R r, C c, W w, E e) {
    return e(r, c, w);
  }

  template <typename E = EdgeOp, typename R = row_value_type, typename C = col_value_type,
            typename W = weight_type>
  __device__
  std::enable_if_t<
    is_valid_edge_op<typename std::result_of<E(R, C)>>::valid,
    typename std::result_of<E(R, C)>::type
  >
  compute(R r, C c, W w, E e) {
    return e(r, c);
  }
};

}  // namesapce detail
}  // namespace experimental
}  // namespace cugraph
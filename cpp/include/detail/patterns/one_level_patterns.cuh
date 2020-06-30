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

#include <graph.hpp>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType,
          typename GraphType,
          typename VertexValueInputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v(HandleType handle,
                     GraphType const& graph_device_view,
                     VertexValueInputIterator vertex_value_input_first,
                     VertexOp v_op,
                     T init);

template <typename HandleType,
          typename GraphType,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v_with_adj_matrix_col(
  HandleType handle,
  GraphType const& graph_device_view,
  VertexValueInputIterator vertex_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  VertexValueOutputIterator vetex_value_output_first,
  VertexOp v_op,
  T init);

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

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
          typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(HandleType handle,
                            GraphType graph,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first);

template <typename HandleType,
          typename GraphType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(HandleType handle,
                            GraphType graph,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first);

template <typename HandleType,
          typename GraphType,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(HandleType handle,
                            GraphType graph,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first);

template <typename HandleType,
          typename GraphType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(HandleType handle,
                            GraphType graph,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first);

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

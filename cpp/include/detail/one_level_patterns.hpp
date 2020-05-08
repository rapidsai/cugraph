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

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename T>
T reduce_v(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    T init);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename T>
T reduce_v(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    VertexValueInputIterator vertex_value_input_last,
    T init);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename VertexOp, typename T>
T transform_reduce_v(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    VertexOp v_op, T init);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename VertexOp>
typename GraphType::vertex_type count_if_v(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    VertexOp v_op);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp, typename T>
T transform_reduce_v_with_adj_matrix_col(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    VertexValueOutputIterator vetex_value_output_first,
    VertexOp v_op, T init);

template <typename HandleType, typename GraphType,
          typename AdjMatrixColValueInputIterator,
          typename ColOp>
typename GraphType::vertex_type count_if_adj_matrix_col(
    HandleType handle, GraphType graph,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    ColOp col_op);

template <typename HandleType, typename GraphType,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename EdgeOp, typename T>
T transform_reduce_e(
    HandleType handle, GraphType graph,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    EdgeOp e_op, T init);

template <typename HandleType, typename GraphType,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename EdgeOp>
typename GraphType::edge_type count_if_e(
    HandleType handle, GraphType graph,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    EdgeOp e_op);

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

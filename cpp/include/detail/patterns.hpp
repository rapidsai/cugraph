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

// copy (interprocess if OPG)

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator, typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first);

template <typename HandleType, typename GraphType,
          typename VertexIterator,
          typename VertexValueInputIterator, typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(
    HandleType handle, GraphType graph,
    VertexIterator vertex_first, VertexIterator vertex_last,
    VertexValueInputIterator vertex_value_input_first,
    AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator, typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(
    HandleType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first);

template <typename HandleType, typename GraphType,
          typename VertexIterator,
          typename VertexValueInputIterator, typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(
    HandleType handle, GraphType graph,
    VertexIterator vertex_first, VertexIterator vertex_last,
    VertexValueInputIterator vertex_value_input_first,
    AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first);

// 1-level

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename T>
T reduce_v(
    HandelType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    T init);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename T>
T reduce_v(
    HandelType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    VertexValueInputIterator vertex_value_input_last,
    T init);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename VertexOp, typename T>
T transform_reduce_v(
    HandelType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    VertexOp v_op, T init);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator,
          typename VertexOp>
GraphType::vertex_type count_if_v(
    HandelType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    VertexOp v_op);

template <typename HandleType, typename GraphType,
          typename VertexValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp, typename T>
T transform_reduce_v_with_adj_matrix_col(
    HandelType handle, GraphType graph,
    VertexValueInputIterator vertex_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    VertexValueOutputIterator vetex_value_output_first,
    VertexOp v_op, T init);

template <typename HandleType, typename GraphType,
          typename AdjMatrixColValueInputIterator,
          typename ColOp>
GraphType::vertex_type count_if_adj_matrix_col(
    HandelType handle, GraphType graph,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    ColOp col_op);

template <typename HandleType, typename GraphType,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename EdgeOp, typename T>
T transform_reduce_e(
    HandelType handle, GraphType graph,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    EdgeOp e_op, T init);

template <typename HandleType, typename GraphType,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename EdgeOp>
GraphType::edge_type count_if_e(
    HandelType handle, GraphType graph,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    EdgeOp e_op);

// 2-levels

template <typename HandleType, typename GraphType,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename EdgeOp, typename T>
void transform_v_transform_reduce_e(
    HandelType handle, GraphType graph,
    AdjMatrixRowValueIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueIterator adj_matrix_col_value_input_first,
    VertexValueInputIterator vertex_value_input_first,
    VertexValueOutputIterator vertex_value_output_first,
    EdgeOp e_op, T init);

template <typename HandleType, typename GraphType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename RowFrontierType,
          typename EdgeOp, typename ReduceOp, typename VertexOp>
void expand_and_transform_if_v_push_if_e(
    HandelType handle, GraphType graph,
    RowIterator row_first, RowIterator row_last,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_row_value_input_last,
    VertexValueInputIterator vertex_value_input_first,
    VertexValueOutputIterator vertex_value_output_first,
    RowFrontierType row_frontier,
    EdgeOp e_op, ReduceOp reduce_op, VertexOp v_op);

/*
iterating over lower triangular (or upper triangular) : triangle counting
LRB might be necessary if the cost of processing an edge (i, j) is a function of degree(i) and
degree(j) : triangle counting
push-pull switching support (e.g. DOBFS), in this case, we need both
CSR & CSC (trade-off execution time vs memory requirement, unless graph is symmetric)
should I take multi-GPU support as a template argument?
Add bool expensive_check = false ?
cugraph::count_if as a multi-GPU wrapper of thrust::count_if? (for expensive check)
if graph is symmetric, there will be additional optimization opportunities (e.g. in-degree == out-degree)
For BFS, sending a bit vector (for the entire set of dest vertices per partitoin may work better
we can use thrust::set_intersection for triangle counting
think about adding thrust wrappers for reduction functions.
thrust::(); if (opg) { allreduce }; can be cugraph::(), and be more consistant with other APIs that
hide communication inside if opg.
Can I pass nullptr for dummy instead of thrust::make_counting_iterator(0)?
*/

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

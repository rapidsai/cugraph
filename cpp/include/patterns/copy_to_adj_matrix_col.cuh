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

#include <experimental/graph_view.hpp>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cugraph {
namespace experimental {

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix column property
 * variables.
 *
 * This version fills the entire set of graph adjacency matrix column property values. This function
 * is inspired by thrust::copy().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixColValueOutputIterator Type of the iterator for graph adjacency matrix column
 * output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_col_value_output_first Iterator pointing to the adjacency matrix column output
 * property variables for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
 */
template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first)
{
  if (GraphViewType::is_multi_gpu) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_local_adj_matrix_partition_cols());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 vertex_value_input_first,
                 vertex_value_input_first + graph_view.get_number_of_local_vertices(),
                 adj_matrix_col_value_output_first);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix column property
 * variables.
 *
 * This version fills only a subset of graph adjacency matrix column property values. [@p
 * vertex_first, @p vertex_last) specifies the vertices with new values to be copied to graph
 * adjacency matrix column property variables. This function is inspired by thrust::copy().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixColValueOutputIterator Type of the iterator for graph adjacency matrix column
 * output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with new values to be
 * copied. v in [vertex_first, vertex_last) should be distinct (and should belong to this process in
 * multi-GPU), otherwise undefined behavior
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with new values to be copied.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_col_value_output_first Iterator pointing to the adjacency matrix column output
 * property variables for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first)
{
  if (GraphViewType::is_multi_gpu) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_local_adj_matrix_partition_cols());
    auto val_first = thrust::make_permutation_iterator(vertex_value_input_first, vertex_first);
    thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    adj_matrix_col_value_output_first);
  }
}

}  // namespace experimental
}  // namespace cugraph

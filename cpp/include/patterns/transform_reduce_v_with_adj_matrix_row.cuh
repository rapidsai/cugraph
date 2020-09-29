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

#include <raft/handle.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

namespace cugraph {
namespace experimental {

/**
 * @brief Apply an operator to the matching vertex and adjacency matrix row properties and reduce.
 *
 * i'th vertex matches with the i'th row in the graph adjacency matrix. @p v_op takes vertex
 * properties and adjacency matrix row properties for the matching row, and @p v_op outputs are
 * reduced. This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam VertexOp Type of the binary vertex operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_adj_matrix_local_rows().
 * @param v_op Binary operator takes *(@p vertex_value_input_first + i) and *(@p
 * adj_matrix_row_value_input_first + j) (where i and j are set for a vertex and the matching row)
 * and returns a transformed value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p v_op outputs.
 */
template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueInputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v_with_adj_matrix_row(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexValueInputIterator vertex_value_input_first,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  VertexOp v_op,
  T init)
{
  if (GraphViewType::is_multi_gpu) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_adj_matrix_local_rows());
    auto input_first = thrust::make_zip_iterator(
      thrust::make_tuple(vertex_value_input_first, adj_matrix_row_value_input_first));
    auto v_op_wrapper = [v_op] __device__(auto v_and_row_val) {
      return v_op(thrust::get<0>(v_and_row_val), thrust::get<1>(v_and_row_val));
    };
    return thrust::transform_reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    input_first,
                                    input_first + graph_view.get_number_of_local_vertices(),
                                    v_op_wrapper,
                                    init,
                                    thrust::plus<T>());
  }
}

}  // namespace experimental
}  // namespace cugraph

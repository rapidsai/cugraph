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
#include <utilities/host_scalar_comm.cuh>

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
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
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
  T ret{};

  auto vertex_first = graph_view.get_local_vertex_first();
  auto vertex_last  = graph_view.get_local_vertex_last();
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto row_first = graph_view.get_local_adj_matrix_partition_row_first(i);
    auto row_last  = graph_view.get_local_adj_matrix_partition_row_last(i);

    auto range_first = std::max(vertex_first, row_first);
    auto range_last  = std::min(vertex_last, row_last);

    if (range_last > range_first) {
      matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);
      auto row_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                      ? 0
                                      : matrix_partition.get_major_value_start_offset();

      auto input_first  = thrust::make_zip_iterator(thrust::make_tuple(
        vertex_value_input_first + (range_first - vertex_first),
        adj_matrix_row_value_input_first + row_value_input_offset + (range_first - row_first)));
      auto v_op_wrapper = [v_op] __device__(auto v_and_row_val) {
        return v_op(thrust::get<0>(v_and_row_val), thrust::get<1>(v_and_row_val));
      };
      ret +=
        thrust::transform_reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                 input_first,
                                 input_first + (range_last - range_first),
                                 v_op_wrapper,
                                 T{},
                                 thrust::plus<T>());
    }
  }

  if (GraphViewType::is_multi_gpu) {
    ret = host_scalar_allreduce(handle.get_comms(), ret, handle.get_stream());
  }

  return init + ret;
}

}  // namespace experimental
}  // namespace cugraph

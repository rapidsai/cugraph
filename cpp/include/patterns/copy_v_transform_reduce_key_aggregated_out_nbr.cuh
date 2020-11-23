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

#include <cuco/static_map.cuh>

#include <type_traits>

namespace cugraph {
namespace experimental {

/**
 * @brief Iterate over the key-aggregated outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the outgoing edges
 * part) and thrust::copy() (update vertex properties part, take transform_reduce output as copy
 * input).
 * Unlike copy_v_transform_reduce_out_nbr, this function first aggregates outgoing edges by key to
 * support two level reduction for each vertex.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam KeyIterator Type of the iterator for graph adjacency matrix column key values for
 * aggregation.
 * @tparam ValueType Type of the value in (key, value) pairs stored in @p kv_map.
 * @tparam KeyAggregatedEdgeOp Type of the quaternary (or quinary) key-aggregated edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for reduction over the key-aggregated outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param out_nbr_key_first Iterator pointing to the adjacency matrix column key (for aggregation)
 * for the first (inclusive) column (assigned to this process in multi-GPU). `out_nbr_key_last`
 * (exclusive) is deduced as @p out_nbr_key_first + @p
 * graph_view.get_number_of_local_adj_matrix_partition_cols().
 * @param kv_map cuco::static_map object holding (key, value) pairs for the keys pointed by @p
 * out_nbr_key_first + i (where i is in [0,
 * graph_view.get_number_of_local_adj_matrix_partition_rows()))
 * @param key_aggregated_e_op Quinary operator takes edge source, key, aggregated edge weight, *(@p
 * adj_matrix_row_value_input_first + i), and value stored in @p kv_map for the key (where i is in
 * [0, graph_view.get_number_of_local_adj_matrix_partition_rows())) and returns a value to be
 * reduced.
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param init Initial value to be added to the reduced @p key_aggregated_e_op return values for
 * each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename KeyIterator,
          typename ValueType,
          typename KeyAggregatedEdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_key_aggregated_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  KeyIterator out_nbr_key_first,
  cuco::static_map<typename std::iterator_traits<KeyIterator>::value_type, ValueType> const& kv_map,
  KeyAggregatedEdgeOp key_aggregated_e_op,
  ReduceOp reduce_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(std::is_integral<typename std::iterator_traits<KeyIterator>::value_type>::value);

  CUGRAPH_FAIL("unimplemented.");
}

}  // namespace experimental
}  // namespace cugraph

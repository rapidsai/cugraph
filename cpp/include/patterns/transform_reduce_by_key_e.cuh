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
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs to (key, value) pairs.
 *
 * This function is inspired by thrust::transform_reduce() and thrust::reduce_by_key().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value of the value in each (key, value) pair.
 * @tparam KeyIterator Type of the iterator for keys in (key, value) pairs.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), *(@p adj_matrix_row_value_input_first + i), and *(@p adj_matrix_col_value_input_first +
 * j) (where i is in [0, graph_view.get_number_of_local_adj_matrix_partition_rows()) and j is in [0,
 * get_number_of_local_adj_matrix_partition_cols())) and returns a pair of a key and a transformed
 * value to be reduced.
 * @param init Initial value to be added to the value in each transform-reduced (key, value) pair.
 * @param map_key_first Iterator pointing to the first (inclusive) key to be stored in the returned
 * cuco::static_map (which is local to this process in mulit-GPU).
 * @param map_key_last Iterator pointing to the last (exclusive) key to be stored in the returned
 * cuco::static_map (which is local to this process in multi-GPU).
 * @return cuco::static_map Hash-based map of (key, value) pairs for the keys pointed by
 * [map_key_first, map_key_last).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T,
          typename KeyIterator>
cuco::static_map<typename std::iterator_traits<KeyIterator>::value_type, T>
transform_reduce_by_key_e(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                          AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                          EdgeOp e_op,
                          T init,
                          KeyIterator map_key_first,
                          KeyIterator map_key_last)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_integral<typename std::iterator_traits<KeyIterator>::value_type>::value);

  CUGRAPH_FAIL("unimplemented.");

  return cuco::static_map<typename std::iterator_traits<KeyIterator>::value_type, T>();
}

}  // namespace experimental
}  // namespace cugraph

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace cugraph {
namespace experimental {

/**
 * @brief Check any of graph adjacency matrix row properties satisfy the given predicate.
 *
 * Returns true if @p row_op returns true for at least once (in any process in multi-GPU), returns
 * false otherwise. This function is inspired by thrust::any_of().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam RowOp Type of the unary predicate operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row properties
 * for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param row_op Unary predicate operator that takes *(@p adj_matrix_row_value_input_first + i)
 * (where i = [0, @p graph_view.get_number_of_local_adj_matrix_partition_rows()) and returns either
 * true or false.
 * @return true If the predicate returns true at least once (in any process in multi-GPU).
 * @return false If the predicate never returns true (in any process in multi-GPU).
 */
template <typename GraphViewType, typename AdjMatrixRowValueInputIterator, typename RowOp>
bool any_of_adj_matrix_row(raft::handle_t const& handle,
                           GraphViewType const& graph_view,
                           AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                           RowOp row_op)
{
  // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
  auto count = thrust::count_if(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    adj_matrix_row_value_input_first,
    adj_matrix_row_value_input_first + graph_view.get_number_of_local_adj_matrix_partition_rows(),
    row_op);
  if (GraphViewType::is_multi_gpu) {
    count = host_scalar_allreduce(handle.get_comms(), count, handle.get_stream());
  }
  return (count > 0);
}

}  // namespace experimental
}  // namespace cugraph

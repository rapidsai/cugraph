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

#include <detail/graph_device_view.cuh>
#include <utilities/error.hpp>

#include <raft/handle.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace cugraph {
namespace experimental {
namespace detail {

/**
 * @brief Check any of graph adjacency matrix row properties satisfy the given predicate.
 *
 * Returns true if @p row_op returns true for at least once (in any process in OPG), returns false
 * otherwise. This function is inspired by thrust::any_of().
 *
 * @tparam HandleType Type of the RAFT handle (e.g. for single-GPU or OPG).
 * @tparam GraphType Type of the passed graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam RowOp Type of the unary predicate operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_device_view Graph object. This graph object should support pass-by-value to device
 * kernels.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row properties
 * for the first (inclusive) row (assigned to this process in OPG).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_device_view.get_number_of_adj_matrix_local_rows().
 * @param row_op Unary predicate operator that takes *(@p adj_matrix_row_value_input_first + i)
 * (where i = [0, @p graph_device_view.get_number_of_adj_matrix_local_rows()) and returns either
 * true or false.
 * @return true If the predicate returns true at least once (in any process in OPG).
 * @return false If the predicate never returns true (in any process in OPG).
 */
template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename RowOp>
bool any_of_adj_matrix_row(HandleType& handle,
                           GraphType const& graph_device_view,
                           AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                           RowOp row_op)
{
  // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
  auto count = thrust::count_if(
    thrust::cuda::par.on(handle.get_stream()),
    adj_matrix_row_value_input_first,
    adj_matrix_row_value_input_first + graph_device_view.get_number_of_adj_matrix_local_rows(),
    row_op);
  if (GraphType::is_opg) {
    // need to reduce count
    CUGRAPH_FAIL("unimplemented.");
  }
  return (count > 0);
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

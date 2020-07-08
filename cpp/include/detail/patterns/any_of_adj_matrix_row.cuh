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

template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename RowOp>
bool any_of_adj_matrix_row(
  HandleType& handle,
  GraphType const& graph_device_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  RowOp row_op)
{
  // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
  auto count = thrust::count_if(thrust::cuda::par.on(handle.get_stream()),
                                adj_matrix_row_value_input_first,
                                adj_matrix_row_value_input_first +
                                  graph_device_view.get_number_of_adj_matrix_local_rows(),
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

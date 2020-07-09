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

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cugraph {
namespace experimental {
namespace detail {

/**
 * @brief
 *
 * @tparam HandleType
 * @tparam GraphType
 * @tparam VertexValueInputIterator
 * @tparam AdjMatrixRowValueOutputIterator
 * @param handle
 * @param graph_device_view
 * @param vertex_value_input_first
 * @param adj_matrix_row_value_output_first
 */
template <typename HandleType,
          typename GraphType,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(HandleType& handle,
                            GraphType const& graph_device_view,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first)
{
  if (GraphType::is_opg) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_rows());
    thrust::copy(thrust::cuda::par.on(handle.get_stream()),
                 vertex_value_input_first,
                 vertex_value_input_first + graph_device_view.get_number_of_local_vertices(),
                 adj_matrix_row_value_output_first);
  }
}

/**
 * @brief
 *
 * @tparam HandleType
 * @tparam GraphType
 * @tparam VertexIterator
 * @tparam VertexValueInputIterator
 * @tparam AdjMatrixRowValueOutputIterator
 * @param handle
 * @param graph_device_view
 * @param vertex_first v in [vertex_first, vertex_last) should be distinct, otherwise undefined
 * behavior
 * @param vertex_last
 * @param vertex_value_input_first
 * @param adj_matrix_row_value_output_first
 */
template <typename HandleType,
          typename GraphType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(HandleType& handle,
                            GraphType const& graph_device_view,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first)
{
  if (GraphType::is_opg) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_rows());
    auto val_first = thrust::make_permutation_iterator(vertex_value_input_first, vertex_first);
    thrust::scatter(thrust::cuda::par.on(handle.get_stream()),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    adj_matrix_row_value_output_first);
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

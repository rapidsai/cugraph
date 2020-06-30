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

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType,
          typename GraphType,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueInputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v_with_adj_matrix_row(
  HandleType& handle,
  GraphType const& graph_device_view,
  VertexValueInputIterator vertex_value_input_first,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  VertexOp v_op,
  T init)
{
  if (GraphType::is_opg) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_rows());
    auto input_first = thrust::make_zip_iterator(
      thrust::make_tuple(vertex_value_input_first, adj_matrix_row_value_input_first));
    auto v_op_wrapper = [v_op] __device__(auto v_and_row_val) {
      return v_op(thrust::get<0>(v_and_row_val), thrust::get<1>(v_and_row_val));
    };
    return thrust::transform_reduce(thrust::cuda::par.on(handle.get_stream()),
                                    input_first,
                                    input_first + graph_device_view.get_number_of_local_vertices(),
                                    v_op_wrapper,
                                    init,
                                    thrust::plus<T>());
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

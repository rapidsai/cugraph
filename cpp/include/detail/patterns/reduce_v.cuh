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

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType, typename GraphType, typename VertexValueInputIterator, typename T>
T reduce_v(HandleType& handle,
           GraphType const& graph_device_view,
           VertexValueInputIterator vertex_value_input_first,
           T init)
{
  auto ret = thrust::reduce(
    thrust::cuda::par.on(handle.get_stream()),
    vertex_value_input_first,
    vertex_value_input_first + graph_device_view.get_number_of_local_vertices(),
    init);
  if (GraphType::is_opg) {
    // need to reduce ret
    CUGRAPH_FAIL("unimplemented.");
  }
  return ret;
}

template <typename HandleType, typename GraphType, typename VertexValueInputIterator, typename T>
T reduce_v(HandleType& handle,
           GraphType const& graph_device_view,
           VertexValueInputIterator vertex_value_input_first,
           VertexValueInputIterator vertex_value_input_last,
           T init)
{
  auto ret = thrust::reduce(thrust::cuda::par.on(handle.get_stream()),
                            vertex_value_input_first,
                            vertex_value_input_last,
                            init);
  if (GraphType::is_opg) {
    // need to reduce ret
    CUGRAPH_FAIL("unimplemented.");
  }
  return ret;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

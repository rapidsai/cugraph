/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename VertexValueInputIterator, typename VertexOp>
struct count_if_call_v_op_t {
  vertex_t local_vertex_partition_range_first{};
  VertexValueInputIterator vertex_value_input_first{};
  VertexOp v_op{};

  __device__ bool operator()(vertex_t i)
  {
    return v_op(local_vertex_partition_range_first + i, *(vertex_value_input_first + i))
             ? vertex_t{1}
             : vertex_t{0};
  }
};

}  // namespace detail

/**
 * @brief Count the number of vertices that satisfies the given predicate.
 *
 * This version iterates over the entire set of graph vertices. This function is inspired by
 * thrust::count_if().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam VertexOp Type of the unary predicate operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param v_op Binary operator takes vertex ID and *(@p vertex_value_input_first + i) (where i is
 * [0, @p graph_view.local_vertex_partition_range_size())) and returns true if this vertex should be
 * included in the returned count.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return GraphViewType::vertex_type Number of times @p v_op returned true.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename VertexOp>
typename GraphViewType::vertex_type count_if_v(raft::handle_t const& handle,
                                               GraphViewType const& graph_view,
                                               VertexValueInputIterator vertex_value_input_first,
                                               VertexOp v_op,
                                               bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto it = thrust::make_transform_iterator(
    thrust::make_counting_iterator(vertex_t{0}),
    detail::count_if_call_v_op_t<vertex_t, VertexValueInputIterator, VertexOp>{
      graph_view.local_vertex_partition_range_first(), vertex_value_input_first, v_op});
  auto count = thrust::reduce(handle.get_thrust_policy(),
                              it,
                              it + graph_view.local_vertex_partition_range_size(),
                              vertex_t{0});
  if (GraphViewType::is_multi_gpu) {
    count =
      host_scalar_allreduce(handle.get_comms(), count, raft::comms::op_t::SUM, handle.get_stream());
  }
  return count;
}

}  // namespace cugraph

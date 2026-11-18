/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename vertex_t, typename VertexValueInputWrapper, typename VertexOp>
struct count_if_call_v_op_t {
  vertex_t local_vertex_partition_range_first;
  VertexValueInputWrapper vertex_value_input;
  VertexOp v_op;

  __device__ bool operator()(vertex_t v_offset)
  {
    return v_op(local_vertex_partition_range_first + v_offset, vertex_value_input.get(v_offset))
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
 * @tparam VertexValueInputWrapper Type of the wrapper for vertex property values.
 * @tparam VertexOp Type of the unary predicate operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input Wrapper used to access vertex property values (for the vertices
 * assigned to this process in multi-GPU).
 * @param v_op Binary operator takes vertex ID and its property value and returns true if this
 * vertex should be included in the returned count.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return GraphViewType::vertex_type Number of times @p v_op returned true.
 */
template <typename GraphViewType, typename VertexValueInputWrapper, typename VertexOp>
typename GraphViewType::vertex_type count_if_v(raft::handle_t const& handle,
                                               GraphViewType const& graph_view,
                                               VertexValueInputWrapper vertex_value_input,
                                               VertexOp v_op,
                                               bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto vertex_value_input_device_view = detail::vertex_property_device_view_t{vertex_value_input};

  auto it = cuda::make_transform_iterator(
    thrust::make_counting_iterator(vertex_t{0}),
    detail::count_if_call_v_op_t<vertex_t, decltype(vertex_value_input_device_view), VertexOp>{
      graph_view.local_vertex_partition_range_first(), vertex_value_input_device_view, v_op});
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

}  // namespace CUGRAPH_EXPORT cugraph

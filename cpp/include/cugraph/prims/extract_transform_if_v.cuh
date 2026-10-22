/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename vertex_t, typename VertexValueInputDeviceView, typename VertexOp, typename T>
struct extract_transform_if_call_v_op_t {
  VertexValueInputDeviceView vertex_value_input{};
  VertexOp v_op{};
  vertex_t v_first{};

  __device__ T operator()(vertex_t offset) const
  {
    return v_op(v_first + offset, vertex_value_input.get(offset));
  }
};

}  // namespace detail

/**
 * @brief Iterate over the entire set of vertices and extract the @p v_op outputs of the vertices
 * that satisfy the given predicate.
 *
 * This function is inspired by thrust::copy_if().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputWrapper Type of the wrapper for input vertex property values.
 * @tparam VertexOp Type of the binary vertex operator.
 * @tparam PredOp Type of the binary predicate operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input Wrapper used to access input vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::view()`.
 * @param v_op Binary operator takes vertex ID and its property value and returns an output value to
 * be extracted.
 * @param pred_op Binary predicate operator takes vertex ID and its property value and returns true
 * if the vertex should be extracted.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Dataframe buffer object storing the extracted @p v_op return values (for the vertices
 * assigned to this process in multi-GPU) in the vertex order.
 */
template <typename GraphViewType,
          typename VertexValueInputWrapper,
          typename VertexOp,
          typename PredOp>
dataframe_buffer_type_t<std::invoke_result_t<VertexOp,
                                             typename GraphViewType::vertex_type,
                                             typename VertexValueInputWrapper::value_type>>
extract_transform_if_v(raft::handle_t const& handle,
                       GraphViewType const& graph_view,
                       VertexValueInputWrapper vertex_value_input,
                       VertexOp v_op,
                       PredOp pred_op,
                       bool do_expensive_check = false)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using v_op_result_t = std::invoke_result_t<VertexOp,
                                             typename GraphViewType::vertex_type,
                                             typename VertexValueInputWrapper::value_type>;
  static_assert(!std::is_same_v<v_op_result_t, void>);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto vertex_value_input_device_view = detail::vertex_property_device_view_t{vertex_value_input};

  auto value_buffer = allocate_dataframe_buffer<v_op_result_t>(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());
  auto value_first = cuda::make_transform_iterator(
    thrust::make_counting_iterator(vertex_t{0}),
    detail::extract_transform_if_call_v_op_t<vertex_t,
                                             decltype(vertex_value_input_device_view),
                                             VertexOp,
                                             v_op_result_t>{
      vertex_value_input_device_view, v_op, graph_view.local_vertex_partition_range_first()});
  auto pred_first = cuda::make_transform_iterator(
    thrust::make_counting_iterator(vertex_t{0}),
    detail::extract_transform_if_call_v_op_t<vertex_t,
                                             decltype(vertex_value_input_device_view),
                                             PredOp,
                                             bool>{
      vertex_value_input_device_view, pred_op, graph_view.local_vertex_partition_range_first()});
  auto last = thrust::copy_if(handle.get_thrust_policy(),
                              value_first,
                              value_first + graph_view.local_vertex_partition_range_size(),
                              pred_first,
                              get_dataframe_buffer_begin(value_buffer),
                              cuda::std::identity{});
  resize_dataframe_buffer(
    value_buffer,
    static_cast<size_t>(cuda::std::distance(get_dataframe_buffer_begin(value_buffer), last)),
    handle.get_stream());
  shrink_to_fit_dataframe_buffer(value_buffer, handle.get_stream());

  return value_buffer;
}

}  // namespace CUGRAPH_EXPORT cugraph

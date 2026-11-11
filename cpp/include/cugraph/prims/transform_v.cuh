/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename vertex_t, typename VertexOp>
struct transform_call_v_op_t {
  vertex_t v_first{};
  VertexOp v_op{};

  __device__ auto operator()(vertex_t offset) const { return v_op(v_first + offset); }
};

}  // namespace detail

/**
 * @brief Transform input vertex property values and write the results to an output vertex property.
 *
 * This function is inspired by thrust::transform(). It updates in place when @p vertex_value_input
 * and @p vertex_value_output are views of the same `vertex_property_t`.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputWrapper Type of the wrapper for input vertex property values.
 * @tparam VertexOp Type of the unary operator.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input Wrapper used to access input vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::view()`.
 * @param v_op Unary operator that takes an input vertex property value and returns the output
 * vertex property value.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexValueInputWrapper,
          typename VertexOp,
          typename VertexValueOutputWrapper>
void transform_v(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 VertexValueInputWrapper vertex_value_input,
                 VertexOp v_op,
                 VertexValueOutputWrapper vertex_value_output,
                 bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto input_value_first  = vertex_value_input.value_first();
  auto output_value_first = vertex_value_output.value_first();
  if constexpr (cugraph::has_packed_bool_element<typename VertexValueOutputWrapper::value_iterator,
                                                 typename VertexValueOutputWrapper::value_type>()) {
    auto num_bools  = static_cast<size_t>(graph_view.local_vertex_partition_range_size());
    auto bool_first = thrust::make_transform_iterator(input_value_first, v_op);
    thrust::tabulate(handle.get_thrust_policy(),
                     output_value_first,
                     output_value_first + packed_bool_size(num_bools),
                     detail::pack_bool_t<decltype(bool_first)>{bool_first, num_bools});
  } else {
    thrust::transform(handle.get_thrust_policy(),
                      input_value_first,
                      input_value_first + graph_view.local_vertex_partition_range_size(),
                      output_value_first,
                      v_op);
  }
}

/**
 * @brief Transform vertex IDs and write the results to an output vertex property.
 *
 * This function is inspired by thrust::transform().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexOp Type of the unary operator.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param v_op Unary operator that takes a vertex ID and returns the output vertex property value.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexOp, typename VertexValueOutputWrapper>
void transform_v(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 VertexOp v_op,
                 VertexValueOutputWrapper vertex_value_output,
                 bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto output_value_first = vertex_value_output.value_first();
  if constexpr (cugraph::has_packed_bool_element<typename VertexValueOutputWrapper::value_iterator,
                                                 typename VertexValueOutputWrapper::value_type>()) {
    auto num_bools = static_cast<size_t>(graph_view.local_vertex_partition_range_size());
    auto bool_first =
      thrust::make_transform_iterator(thrust::make_counting_iterator(vertex_t{0}),
                                      detail::transform_call_v_op_t<vertex_t, VertexOp>{
                                        graph_view.local_vertex_partition_range_first(), v_op});
    thrust::tabulate(handle.get_thrust_policy(),
                     output_value_first,
                     output_value_first + packed_bool_size(num_bools),
                     detail::pack_bool_t<decltype(bool_first)>{bool_first, num_bools});
  } else {
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(vertex_t{0}) + graph_view.local_vertex_partition_range_size(),
      output_value_first,
      detail::transform_call_v_op_t<vertex_t, VertexOp>{
        graph_view.local_vertex_partition_range_first(), v_op});
  }
}

}  // namespace CUGRAPH_EXPORT cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace CUGRAPH_EXPORT cugraph {

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
  thrust::transform(handle.get_thrust_policy(),
                    input_value_first,
                    input_value_first + graph_view.local_vertex_partition_range_size(),
                    output_value_first,
                    v_op);
}

}  // namespace CUGRAPH_EXPORT cugraph

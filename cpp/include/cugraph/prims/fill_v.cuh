/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Fill vertex property values with the input value.
 *
 * This function is inspired by thrust::fill().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @tparam T Type of the vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param input Vertex property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexValueOutputWrapper, typename T>
void fill_v(raft::handle_t const& handle,
            GraphViewType const& graph_view,
            VertexValueOutputWrapper vertex_value_output,
            T input,
            bool do_expensive_check = false)
{
  static_assert(std::is_same_v<T, typename VertexValueOutputWrapper::value_type>);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto output_value_first = vertex_value_output.value_first();
  thrust::fill(handle.get_thrust_policy(),
               output_value_first,
               output_value_first + graph_view.local_vertex_partition_range_size(),
               input);
}

}  // namespace CUGRAPH_EXPORT cugraph

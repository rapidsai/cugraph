/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/core/handle.hpp>

#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Fill vertex property values with a sequence of values (@p init, @p init + @p step, @p
 * init + 2 * @p step, ...).
 *
 * This function is inspired by thrust::sequence().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @tparam T Type of the vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param init The value of the first vertex in the local vertex partition range.
 * @param step The difference between the values of two consecutive vertices.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexValueOutputWrapper, typename T>
void sequence_v(raft::handle_t const& handle,
                GraphViewType const& graph_view,
                VertexValueOutputWrapper vertex_value_output,
                T init,
                T step                  = T{1},
                bool do_expensive_check = false)
{
  static_assert(std::is_same_v<T, typename VertexValueOutputWrapper::value_type>);
  static_assert(!cugraph::has_packed_bool_element<typename VertexValueOutputWrapper::value_iterator,
                                                  typename VertexValueOutputWrapper::value_type>(),
                "sequence_v does not support bool vertex property values.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto output_value_first = vertex_value_output.value_first();
  cugraph::sequence(handle.get_thrust_policy(),
                    output_value_first,
                    output_value_first + graph_view.local_vertex_partition_range_size(),
                    init,
                    step);
}

}  // namespace CUGRAPH_EXPORT cugraph

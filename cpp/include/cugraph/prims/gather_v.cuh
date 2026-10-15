/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/distance.h>
#include <thrust/transform.h>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Gather vertex property values for the input vertices.
 *
 * This function is inspired by thrust::gather(). For each vertex in [@p vertex_first, @p
 * vertex_last), the corresponding vertex property value is written to @p output_first. Vertex IDs
 * are local to this GPU.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator Type of the iterator for input vertex IDs.
 * @tparam VertexValueInputWrapper Type of the wrapper for input vertex property values.
 * @tparam OutputIterator Type of the iterator for gathered output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Beginning of the vertex IDs to gather.
 * @param vertex_last End of the vertex IDs to gather.
 * @param vertex_value_input Wrapper used to access input vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::view()`.
 * @param output_first Beginning of the output range. The output range must hold at least
 * `thrust::distance(vertex_first, vertex_last)` elements.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputWrapper,
          typename OutputIterator>
void gather_v(raft::handle_t const& handle,
              GraphViewType const& graph_view,
              VertexIterator vertex_first,
              VertexIterator vertex_last,
              VertexValueInputWrapper vertex_value_input,
              OutputIterator output_first,
              bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (cugraph::has_packed_bool_element<typename VertexValueInputWrapper::value_iterator,
                                                 typename VertexValueInputWrapper::value_type>()) {
    auto device_view = detail::vertex_property_device_view_t{vertex_value_input};
    thrust::transform(
      handle.get_thrust_policy(),
      vertex_first,
      vertex_last,
      output_first,
      [device_view, v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        return device_view.get(v - v_first);
      });
  } else {
    auto map_first = cuda::make_transform_iterator(
      vertex_first,
      detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()});
    cugraph::gather(handle.get_thrust_policy(),
                    map_first,
                    map_first + thrust::distance(vertex_first, vertex_last),
                    vertex_value_input.value_first(),
                    output_first);
  }
}

}  // namespace CUGRAPH_EXPORT cugraph

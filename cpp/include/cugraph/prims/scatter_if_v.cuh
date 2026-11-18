/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/vertex_partition_device_view.cuh>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Conditionally scatter input values to a vertex property.
 *
 * This function is inspired by thrust::scatter_if(). The value at position `i` in [@p input_first,
 * @p input_last) is written to the vertex property of the vertex at position `i` in the range
 * beginning at @p vertex_first when @p pred returns true for the value at position `i` in the range
 * beginning at @p stencil_first. Vertex IDs are local to this GPU.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam InputIterator Type of the iterator for input values.
 * @tparam VertexIterator Type of the iterator for input vertex IDs.
 * @tparam StencilIterator Type of the iterator for stencil values.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @tparam Predicate Type of the unary predicate applied to stencil values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input_first Beginning of the values to scatter.
 * @param input_last End of the values to scatter.
 * @param vertex_first Beginning of the vertex IDs. This range must hold at least
 * `thrust::distance(input_first, input_last)` vertices.
 * @param stencil_first Beginning of the stencil values. This range must hold at least
 * `thrust::distance(input_first, input_last)` values.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param pred Unary predicate applied to each stencil value. The corresponding input value is
 * scattered only when this returns true.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename InputIterator,
          typename VertexIterator,
          typename StencilIterator,
          typename VertexValueOutputWrapper,
          typename Predicate>
void scatter_if_v(raft::handle_t const& handle,
                  GraphViewType const& graph_view,
                  InputIterator input_first,
                  InputIterator input_last,
                  VertexIterator vertex_first,
                  StencilIterator stencil_first,
                  VertexValueOutputWrapper vertex_value_output,
                  Predicate pred,
                  bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (cugraph::has_packed_bool_element<typename VertexValueOutputWrapper::value_iterator,
                                                 typename VertexValueOutputWrapper::value_type>()) {
    auto num_values       = static_cast<size_t>(thrust::distance(input_first, input_last));
    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      graph_view.local_vertex_partition_view());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(num_values),
                     [device_view = detail::vertex_property_device_view_t{vertex_value_output},
                      vertex_partition,
                      input_first,
                      vertex_first,
                      stencil_first,
                      pred] __device__(size_t i) {
                       if (pred(*(stencil_first + i))) {
                         auto v = *(vertex_first + i);
                         auto v_offset =
                           vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                         device_view.set(v_offset, *(input_first + i));
                       }
                     });
  } else {
    auto map_first = cuda::make_transform_iterator(
      vertex_first,
      detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()});
    thrust::scatter_if(handle.get_thrust_policy(),
                       input_first,
                       input_last,
                       map_first,
                       stencil_first,
                       vertex_value_output.value_first(),
                       pred);
  }
}

/**
 * @brief Conditionally scatter input values to a vertex property.
 *
 * This function is inspired by thrust::scatter_if(). The value at position `i` in [@p input_first,
 * @p input_last) is written to the vertex property of the vertex at position `i` in the range
 * beginning at @p vertex_first when @p pred returns true for that input value. Vertex IDs are local
 * to this GPU.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam InputIterator Type of the iterator for input values.
 * @tparam VertexIterator Type of the iterator for input vertex IDs.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @tparam Predicate Type of the unary predicate applied to input values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input_first Beginning of the values to scatter.
 * @param input_last End of the values to scatter.
 * @param vertex_first Beginning of the vertex IDs. This range must hold at least
 * `thrust::distance(input_first, input_last)` vertices.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param pred Unary predicate applied to each input value. The input value is scattered only when
 * this returns true.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename InputIterator,
          typename VertexIterator,
          typename VertexValueOutputWrapper,
          typename Predicate>
void scatter_if_v(raft::handle_t const& handle,
                  GraphViewType const& graph_view,
                  InputIterator input_first,
                  InputIterator input_last,
                  VertexIterator vertex_first,
                  VertexValueOutputWrapper vertex_value_output,
                  Predicate pred,
                  bool do_expensive_check = false)
{
  scatter_if_v(handle,
               graph_view,
               input_first,
               input_last,
               vertex_first,
               input_first,
               vertex_value_output,
               pred,
               do_expensive_check);
}

}  // namespace CUGRAPH_EXPORT cugraph

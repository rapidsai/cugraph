/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <cstddef>

namespace cugraph {

namespace detail {

template <typename GraphViewType, typename T, typename EdgeMajorPropertyOutputWrapper>
void fill_edge_major_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              T input,
                              EdgeMajorPropertyOutputWrapper edge_major_property_output)
{
  auto keys         = edge_major_property_output.keys();
  auto value_firsts = edge_major_property_output.value_firsts();
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    size_t buffer_size{0};
    if (keys) {
      buffer_size = (*keys)[i].size();
    } else {
      if constexpr (GraphViewType::is_storage_transposed) {
        buffer_size = static_cast<size_t>(graph_view.local_edge_partition_dst_range_size(i));
      } else {
        buffer_size = static_cast<size_t>(graph_view.local_edge_partition_src_range_size(i));
      }
    }
    thrust::fill_n(handle.get_thrust_policy(), value_firsts[i], buffer_size, input);
  }
}

template <typename GraphViewType, typename T, typename EdgeMinorPropertyOutputWrapper>
void fill_edge_minor_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              T input,
                              EdgeMinorPropertyOutputWrapper edge_minor_property_output)
{
  auto keys = edge_minor_property_output.keys();
  size_t buffer_size{0};
  if (keys) {
    buffer_size = (*keys).size();
  } else {
    if constexpr (GraphViewType::is_storage_transposed) {
      buffer_size = static_cast<size_t>(graph_view.local_edge_partition_src_range_size());
    } else {
      buffer_size = static_cast<size_t>(graph_view.local_edge_partition_dst_range_size());
    }
  }
  thrust::fill_n(
    handle.get_thrust_policy(), edge_minor_property_output.value_first(), buffer_size, input);
}

}  // namespace detail

/**
 * @brief Fill graph edge source property values to the input value.
 *
 * This version fills graph edge source property values for the entire edge source ranges (assigned
 * to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge source property values will be set to @p input.
 * @param edge_src_property_output edge_src_property_t class object to store source property values
 * (for the edge source assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename T>
void fill_edge_src_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            T input,
                            edge_src_property_t<GraphViewType, T>& edge_src_property_output,
                            bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::fill_edge_minor_property(
      handle, graph_view, input, edge_src_property_output.mutable_view());
  } else {
    detail::fill_edge_major_property(
      handle, graph_view, input, edge_src_property_output.mutable_view());
  }
}

/**
 * @brief Fill graph edge destination property values to the input value.
 *
 * This version fills graph edge destination property values for the entire edge destination ranges
 * (assigned to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge destination property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge destination property values will be set to @p input.
 * @param edge_dst_property_output edge_dst_property_t class object to store destination property
 * values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename T>
void fill_edge_dst_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            T input,
                            edge_dst_property_t<GraphViewType, T>& edge_dst_property_output,
                            bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::fill_edge_major_property(
      handle, graph_view, input, edge_dst_property_output.mutable_view());
  } else {
    detail::fill_edge_minor_property(
      handle, graph_view, input, edge_dst_property_output.mutable_view());
  }
}

}  // namespace cugraph

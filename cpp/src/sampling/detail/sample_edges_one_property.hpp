/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {

template <typename vertex_t, typename tag_t, bool multi_gpu, bool sorted_unique>
class key_bucket_view_t;

namespace detail {

/**
 * @brief Randomly sample outgoing edges with a single edge property view.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam property_view_t Type of the edge property view passed to the sampling primitive.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 */
template <typename vertex_t, typename edge_t, typename property_view_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
sample_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false> const& key_bucket_view,
  raft::host_span<size_t const> Ks,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  bool with_replacement);

/**
 * @brief Randomly sample unvisited outgoing edges with a single edge property view.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam tag_t Type of the key bucket tag.
 * @tparam property_view_t Type of the edge property view passed to the sampling primitive.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 */
template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename property_view_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<int32_t>>>
sample_unvisited_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false> const& key_bucket_view,
  rmm::device_uvector<vertex_t>&& visited_minors,
  std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,
  raft::host_span<size_t const> Ks,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  bool with_replacement);

/**
 * @brief Randomly sample temporal outgoing edges with a single edge property view.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam property_view_t Type of the edge property view passed to the sampling primitive.
 * @tparam time_stamp_t Type of time. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 */
template <typename vertex_t,
          typename edge_t,
          typename property_view_t,
          typename time_stamp_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::key_bucket_view_t<vertex_t, time_stamp_t, multi_gpu, false> const& key_bucket_view,
  raft::host_span<size_t const> Ks,
  bool with_replacement,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<time_stamp_t const> active_major_times,
  std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,
  temporal_sampling_comparison_t temporal_sampling_comparison);

}  // namespace detail
}  // namespace cugraph

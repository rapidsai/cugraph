/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "neighbor_sampling_impl.cuh"
#include "temporal_sampling_impl.cuh"

#include <cugraph/export.hpp>
#include <cugraph/sampling_functions.hpp>

#include <optional>
#include <tuple>

namespace cugraph {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          bool store_transposed,
          bool multi_gpu>
CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,
                          rmm::device_uvector<vertex_t>,
                          std::optional<rmm::device_uvector<weight_t>>,
                          std::optional<rmm::device_uvector<edge_t>>,
                          std::optional<rmm::device_uvector<edge_type_t>>,
                          std::optional<rmm::device_uvector<time_stamp_t>>,
                          std::optional<rmm::device_uvector<time_stamp_t>>,
                          std::optional<rmm::device_uvector<int32_t>>,
                          std::optional<rmm::device_uvector<size_t>>>
neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  std::optional<edge_type_t> num_edge_types,
  sampling_options_t sampling_options,
  bool do_expensive_check)
{
  auto const is_temporal        = sampling_options.temporal_sampling_comparison.has_value();
  auto const neighbor_selection = sampling_options.neighbor_selection;

  CUGRAPH_EXPECTS(!num_edge_types || edge_type_view,
                  "edge_type_view is required when num_edge_types is specified.");
  CUGRAPH_EXPECTS(is_temporal == edge_start_time_view.has_value(),
                  "edge_start_time_view and temporal_sampling_comparison must either both be "
                  "specified or both be absent.");
  CUGRAPH_EXPECTS(is_temporal || (!edge_end_time_view && !starting_vertex_start_times &&
                                  !starting_vertex_end_times),
                  "Temporal edge and seed time arguments require temporal_sampling_comparison.");
  CUGRAPH_EXPECTS(neighbor_selection == neighbor_selection_t::RANDOM || is_temporal,
                  "LAST neighbor selection requires temporal sampling.");
  CUGRAPH_EXPECTS(neighbor_selection == neighbor_selection_t::RANDOM || !edge_bias_view,
                  "LAST neighbor selection does not accept edge biases.");
  CUGRAPH_EXPECTS(
    neighbor_selection == neighbor_selection_t::RANDOM || !sampling_options.with_replacement,
    "LAST neighbor selection does not support sampling with replacement.");
  if (neighbor_selection != neighbor_selection_t::RANDOM) {
    CUGRAPH_FAIL("LAST neighbor selection is not yet implemented.");
  }
  CUGRAPH_EXPECTS(!(sampling_options.with_replacement && sampling_options.disjoint_sampling),
                  "Invalid input argument: disjoint sampling and sampling with replacement are "
                  "mutually exclusive.");
  CUGRAPH_EXPECTS(
    !is_temporal || sampling_options.disjoint_sampling,
    "Invalid input argument: temporal neighbor sampling requires disjoint sampling; set "
    "sampling_options.disjoint_sampling to true.");

  if (is_temporal) {
    return detail::temporal_neighbor_sample_impl<vertex_t,
                                                 edge_t,
                                                 weight_t,
                                                 edge_type_t,
                                                 time_stamp_t,
                                                 weight_t>(handle,
                                                           rng_state,
                                                           graph_view,
                                                           edge_weight_view,
                                                           edge_id_view,
                                                           edge_type_view,
                                                           *edge_start_time_view,
                                                           edge_end_time_view,
                                                           edge_bias_view,
                                                           starting_vertices,
                                                           starting_vertex_start_times,
                                                           starting_vertex_end_times,
                                                           starting_vertex_labels,
                                                           label_to_output_comm_rank,
                                                           fan_out,
                                                           num_edge_types,
                                                           sampling_options,
                                                           do_expensive_check);
  }

  rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> hops{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  std::tie(srcs, dsts, weights, edge_ids, edge_types, hops, std::ignore, offsets) =
    detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, weight_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      edge_type_view,
      edge_bias_view,
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      num_edge_types,
      sampling_options,
      do_expensive_check);

  return std::make_tuple(std::move(srcs),
                         std::move(dsts),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::optional<rmm::device_uvector<time_stamp_t>>{std::nullopt},
                         std::optional<rmm::device_uvector<time_stamp_t>>{std::nullopt},
                         std::move(hops),
                         std::move(offsets));
}

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "traversal/od_shortest_distances_impl.cuh"

namespace cugraph {

// SG instantiation

template rmm::device_uvector<float> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view,
  raft::device_span<int64_t const> origins,
  raft::device_span<int64_t const> destinations,
  float cutoff,
  bool do_expensive_check);

template rmm::device_uvector<double> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view,
  raft::device_span<int64_t const> origins,
  raft::device_span<int64_t const> destinations,
  double cutoff,
  bool do_expensive_check);

}  // namespace cugraph

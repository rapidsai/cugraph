/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/leiden_impl.cuh"

namespace cugraph {

// SG instantiation

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  size_t max_level,
  float resolution,
  float theta);

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  size_t max_level,
  double resolution,
  double theta);

template std::pair<size_t, float> leiden(raft::handle_t const&,
                                         raft::random::RngState&,
                                         graph_view_t<int32_t, int32_t, false, true> const&,
                                         std::optional<edge_property_view_t<int32_t, float const*>>,
                                         int32_t*,
                                         size_t,
                                         float,
                                         float);
template std::pair<size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  int32_t*,
  size_t,
  double,
  double);
}  // namespace cugraph

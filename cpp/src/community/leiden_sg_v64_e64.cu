/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/leiden_impl.cuh"

namespace cugraph {

// SG instantiation

template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  size_t max_level,
  float resolution,
  float theta);

template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  size_t max_level,
  double resolution,
  double theta);

template std::pair<size_t, float> leiden(raft::handle_t const&,
                                         raft::random::RngState&,
                                         graph_view_t<int64_t, int64_t, false, false> const&,
                                         std::optional<edge_property_view_t<int64_t, float const*>>,
                                         int64_t*,
                                         size_t,
                                         float,
                                         float);
template std::pair<size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int64_t, int64_t, false, false> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  int64_t*,
  size_t,
  double,
  double);

}  // namespace cugraph

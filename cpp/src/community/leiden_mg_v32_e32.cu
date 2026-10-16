/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/leiden_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT std::pair<std::unique_ptr<dendrogram_t<int32_t>>, float> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  size_t max_level,
  float resolution,
  float theta);

template CUGRAPH_EXPORT std::pair<std::unique_ptr<dendrogram_t<int32_t>>, double> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  size_t max_level,
  double resolution,
  double theta);

template CUGRAPH_EXPORT std::pair<size_t, float> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  int32_t*,
  size_t,
  float,
  float);

template CUGRAPH_EXPORT std::pair<size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  int32_t*,
  size_t,
  double,
  double);

template CUGRAPH_EXPORT std::pair<dendrogram_t<int32_t>, float> leiden_dendrogram(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  size_t,
  float,
  float);

template CUGRAPH_EXPORT std::pair<dendrogram_t<int32_t>, double> leiden_dendrogram(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  size_t,
  double,
  double);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int32_t, int32_t>, size_t, float> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  bool,
  size_t,
  float,
  float);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int32_t, int32_t>, size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  bool,
  size_t,
  double,
  double);

}  // namespace cugraph

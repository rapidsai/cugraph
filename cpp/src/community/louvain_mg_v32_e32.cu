/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/louvain_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// Explicit template instantations

template CUGRAPH_EXPORT std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  size_t,
  float,
  float);
template CUGRAPH_EXPORT std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  size_t,
  double,
  double);
template CUGRAPH_EXPORT std::pair<size_t, float> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  int32_t*,
  size_t,
  float,
  float);
template CUGRAPH_EXPORT std::pair<size_t, double> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  int32_t*,
  size_t,
  double,
  double);
}  // namespace cugraph

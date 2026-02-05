/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/louvain_impl.cuh"

namespace cugraph {

// Explicit template instantations

template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int64_t, int64_t, false, false> const&,
  std::optional<edge_property_view_t<int64_t, float const*>>,
  size_t,
  float,
  float);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int64_t, int64_t, false, false> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  size_t,
  double,
  double);

template std::pair<size_t, float> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int64_t, int64_t, false, false> const&,
  std::optional<edge_property_view_t<int64_t, float const*>>,
  int64_t*,
  size_t,
  float,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int64_t, int64_t, false, false> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  int64_t*,
  size_t,
  double,
  double);

}  // namespace cugraph

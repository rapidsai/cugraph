/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "centrality/eigenvector_centrality_impl.cuh"

namespace cugraph {

// SG instantiation

template rmm::device_uvector<float> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<float const>> initial_centralities,
  float epsilon,
  size_t max_iterations,
  bool do_expensive_check);

template rmm::device_uvector<double> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<double const>> initial_centralities,
  double epsilon,
  size_t max_iterations,
  bool do_expensive_check);

}  // namespace cugraph

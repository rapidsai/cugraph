/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "centrality/eigenvector_centrality_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT rmm::device_uvector<float> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<float const>> initial_centralities,
  float epsilon,
  size_t max_iterations,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<double> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<double const>> initial_centralities,
  double epsilon,
  size_t max_iterations,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int32_t, float>,
                                   centrality_algorithm_metadata_t>
eigenvector_centrality(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                       std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                       std::optional<vertex_property_t<int32_t, float>>&& initial_centralities,
                       float epsilon,
                       size_t max_iterations,
                       bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int32_t, double>,
                                   centrality_algorithm_metadata_t>
eigenvector_centrality(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                       std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                       std::optional<vertex_property_t<int32_t, double>>&& initial_centralities,
                       double epsilon,
                       size_t max_iterations,
                       bool do_expensive_check);

}  // namespace cugraph

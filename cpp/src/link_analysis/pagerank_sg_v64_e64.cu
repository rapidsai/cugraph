/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "link_analysis/pagerank_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation
template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<float>, centrality_algorithm_metadata_t>
pagerank(raft::handle_t const& handle,
         graph_view_t<int64_t, int64_t, true, false> const& graph_view,
         std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
         std::optional<raft::device_span<float const>> precomputed_vertex_out_weight_sums,
         std::optional<std::tuple<raft::device_span<int64_t const>, raft::device_span<float const>>>
           personalization,
         std::optional<raft::device_span<float const>> initial_pageranks,
         float alpha,
         float epsilon,
         size_t max_iterations,
         bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<double>, centrality_algorithm_metadata_t>
pagerank(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<double const>> precomputed_vertex_out_weight_sums,
  std::optional<std::tuple<raft::device_span<int64_t const>, raft::device_span<double const>>>
    personalization,
  std::optional<raft::device_span<double const>> initial_pageranks,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool do_expensive_check);

template CUGRAPH_EXPORT
  std::tuple<vertex_property_t<int64_t, float>, iterative_algorithm_metadata_t>
  pagerank(
    raft::handle_t const& handle,
    graph_view_t<int64_t, int64_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
    std::optional<vertex_property_view_t<int64_t, float const*>> precomputed_vertex_out_weight_sums,
    std::optional<std::tuple<raft::device_span<int64_t const>, raft::device_span<float const>>>
      personalization,
    std::optional<vertex_property_t<int64_t, float>>&& initial_pageranks,
    float alpha,
    float epsilon,
    size_t max_iterations,
    bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int64_t, double>,
                                   iterative_algorithm_metadata_t>
pagerank(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<vertex_property_view_t<int64_t, double const*>> precomputed_vertex_out_weight_sums,
  std::optional<std::tuple<raft::device_span<int64_t const>, raft::device_span<double const>>>
    personalization,
  std::optional<vertex_property_t<int64_t, double>>&& initial_pageranks,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool do_expensive_check);

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "community/detail/common_methods.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/fill_edge_property.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t min_weight,
  size_t ensemble_size,
  bool compact_cluster_ids,
  size_t max_level,
  weight_t threshold,
  weight_t resolution)
{
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  CUGRAPH_EXPECTS(min_weight >= weight_t{0.0},
                  "Invalid input arguments: min_weight must be positive");
  CUGRAPH_EXPECTS(ensemble_size >= 1,
                  "Invalid input arguments: ensemble_size must be a non-zero integer");
  CUGRAPH_EXPECTS(
    threshold > 0.0 && threshold <= 1.0,
    "Invalid input arguments: threshold must be a positive number in range (0.0, 1.0]");
  CUGRAPH_EXPECTS(
    resolution > 0.0 && resolution <= 1.0,
    "Invalid input arguments: resolution must be a positive number in range (0.0, 1.0]");

  edge_src_property_t<vertex_t, vertex_t> src_cluster_assignments(handle, graph_view);
  edge_dst_property_t<vertex_t, vertex_t> dst_cluster_assignments(handle, graph_view);
  edge_property_t<edge_t, weight_t> modified_edge_weights(handle, graph_view);

  cugraph::fill_edge_property(
    handle, graph_view, modified_edge_weights.mutable_view(), weight_t{0});

  weight_t modularity = -1.0;
  vertex_property_t<vertex_t, vertex_t> cluster_assignments(handle, graph_view);

  for (size_t i = 0; i < ensemble_size; i++) {
    auto [dendrogram, this_modularity] = cugraph::louvain_dendrogram(
      handle,
      std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
      graph_view,
      edge_weight_view,
      size_t{1},
      threshold,
      resolution);
    modularity          = this_modularity;
    cluster_assignments = cugraph::flatten_dendrogram(handle, graph_view, dendrogram, false);

    cugraph::update_edge_src_property(
      handle, graph_view, cluster_assignments.view(), src_cluster_assignments.mutable_view());
    cugraph::update_edge_dst_property(
      handle, graph_view, cluster_assignments.view(), dst_cluster_assignments.mutable_view());

    cugraph::transform_e(
      handle,
      graph_view,
      src_cluster_assignments.view(),
      dst_cluster_assignments.view(),
      modified_edge_weights.view(),
      [] __device__(auto, auto, auto src_property, auto dst_property, auto edge_property) {
        return edge_property + (src_property == dst_property);
      },
      modified_edge_weights.mutable_view());
  }

  cugraph::transform_e(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    view_concat(*edge_weight_view, modified_edge_weights.view()),
    [min_weight, ensemble_size = static_cast<weight_t>(ensemble_size)] __device__(
      auto, auto, cuda::std::nullopt_t, cuda::std::nullopt_t, auto edge_properties) {
      auto e_weight    = cuda::std::get<0>(edge_properties);
      auto e_frequency = cuda::std::get<1>(edge_properties);
      return min_weight + (e_weight - min_weight) * e_frequency / ensemble_size;
    },
    modified_edge_weights.mutable_view());

  auto [dendrogram, this_modularity] = cugraph::louvain_dendrogram(
    handle,
    std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
    graph_view,
    std::make_optional(modified_edge_weights.view()),
    max_level,
    threshold,
    resolution);
  modularity = this_modularity;
  max_level  = dendrogram.num_levels();
  cluster_assignments =
    cugraph::flatten_dendrogram(handle, graph_view, dendrogram, compact_cluster_ids);

  // Compute final modularity using original edge weights
  weight_t total_edge_weight =
    cugraph::compute_total_edge_weight(handle, graph_view, *edge_weight_view);

  if constexpr (multi_gpu) {
    cugraph::update_edge_src_property(
      handle, graph_view, cluster_assignments.view(), src_cluster_assignments.mutable_view());
    cugraph::update_edge_dst_property(
      handle, graph_view, cluster_assignments.view(), dst_cluster_assignments.mutable_view());
  }

  auto cluster_assignment_ids = cluster_assignments.release_buffer();

  auto [cluster_keys, cluster_weights] = cugraph::detail::compute_cluster_keys_and_values(
    handle, graph_view, edge_weight_view, cluster_assignment_ids, src_cluster_assignments);

  modularity = detail::compute_modularity(handle,
                                          graph_view,
                                          edge_weight_view,
                                          src_cluster_assignments,
                                          dst_cluster_assignments,
                                          cluster_assignment_ids,
                                          cluster_weights,
                                          total_edge_weight,
                                          resolution);

  return std::make_tuple(
    vertex_property_t<vertex_t, vertex_t>(std::move(cluster_assignment_ids),
                                          graph_view.local_vertex_partition_range_size()),
    max_level,
    modularity);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t min_weight,
  size_t ensemble_size,
  bool compact_cluster_ids,
  size_t max_level,
  weight_t threshold,
  weight_t resolution)
{
  return detail::ecg(handle,
                     rng_state,
                     graph_view,
                     edge_weight_view,
                     min_weight,
                     ensemble_size,
                     compact_cluster_ids,
                     max_level,
                     threshold,
                     resolution);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t min_weight,
  size_t ensemble_size,
  size_t max_level,
  weight_t threshold,
  weight_t resolution)
{
  auto [cluster_assignments, num_levels, modularity] = cugraph::ecg(handle,
                                                                    rng_state,
                                                                    graph_view,
                                                                    edge_weight_view,
                                                                    min_weight,
                                                                    ensemble_size,
                                                                    false,
                                                                    max_level,
                                                                    threshold,
                                                                    resolution);

  return std::make_tuple(cluster_assignments.release_buffer(), num_levels, modularity);
}

}  // namespace cugraph

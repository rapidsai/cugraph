/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <prims/fill_edge_property.cuh>
#include <prims/transform_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <community/detail/common_methods.hpp>
#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {

namespace detail {

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
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  edge_src_property_t<graph_view_t, vertex_t> src_cluster_assignments(handle, graph_view);
  edge_dst_property_t<graph_view_t, vertex_t> dst_cluster_assignments(handle, graph_view);
  edge_property_t<graph_view_t, weight_t> modified_edge_weights(handle, graph_view);

  cugraph::fill_edge_property(handle, graph_view, weight_t{0}, modified_edge_weights);

  weight_t modularity = -1.0;
  rmm::device_uvector<vertex_t> cluster_assignments(graph_view.local_vertex_partition_range_size(),
                                                    handle.get_stream());

  for (size_t i = 0; i < ensemble_size; i++) {
    std::tie(std::ignore, modularity) = cugraph::louvain(
      handle,
      std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
      graph_view,
      edge_weight_view,
      cluster_assignments.data(),
      size_t{1},
      threshold,
      resolution);

    cugraph::update_edge_src_property(
      handle, graph_view, cluster_assignments.begin(), src_cluster_assignments);
    cugraph::update_edge_dst_property(
      handle, graph_view, cluster_assignments.begin(), dst_cluster_assignments);

    cugraph::transform_e(
      handle,
      graph_view,
      src_cluster_assignments.view(),
      dst_cluster_assignments.view(),
      modified_edge_weights.view(),
      [] __device__(auto src, auto dst, auto src_property, auto dst_property, auto edge_property) {
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
    [min_weight, ensemble_size] __device__(
      auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto edge_properties) {
      auto e_weight    = thrust::get<0>(edge_properties);
      auto e_frequency = thrust::get<1>(edge_properties);
      return min_weight + (e_weight - min_weight) * e_frequency / ensemble_size;
    },
    modified_edge_weights.mutable_view());

  std::tie(max_level, modularity) =

    cugraph::louvain(handle,
                     std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
                     graph_view,
                     std::make_optional(modified_edge_weights.view()),
                     cluster_assignments.data(),
                     max_level,
                     threshold,
                     resolution);
  // Compute final modularity using original edge weights

  weight_t total_edge_weight =
    cugraph::compute_total_edge_weight(handle, graph_view, *edge_weight_view);

  if constexpr (multi_gpu) {
    cugraph::update_edge_src_property(
      handle, graph_view, cluster_assignments.begin(), src_cluster_assignments);
    cugraph::update_edge_dst_property(
      handle, graph_view, cluster_assignments.begin(), dst_cluster_assignments);
  }

  auto [cluster_keys, cluster_weights] = cugraph::detail::compute_cluster_keys_and_values(
    handle, graph_view, edge_weight_view, cluster_assignments, src_cluster_assignments);

  modularity = detail::compute_modularity(handle,
                                          graph_view,
                                          edge_weight_view,
                                          src_cluster_assignments,
                                          dst_cluster_assignments,
                                          cluster_assignments,
                                          cluster_weights,
                                          total_edge_weight,
                                          resolution);

  return std::make_tuple(std::move(cluster_assignments), max_level, modularity);
}

}  // namespace detail

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
  return detail::ecg(handle,
                     rng_state,
                     graph_view,
                     edge_weight_view,
                     min_weight,
                     ensemble_size,
                     max_level,
                     threshold,
                     resolution);
}

}  // namespace cugraph

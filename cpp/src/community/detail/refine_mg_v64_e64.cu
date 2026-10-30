/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "community/detail/refine_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT
  std::tuple<vertex_property_t<int64_t, int64_t>,
             std::pair<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
  refine_clustering<int64_t, int64_t, true, float>(
    raft::handle_t const& handle,
    raft::random::RngState& rng_state,
    cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
    std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
    vertex_property_view_t<int64_t, float const*> vertex_weights_v,
    vertex_property_view_t<int64_t, int64_t const*> next_clusters_v,
    edge_src_property_t<int64_t, float> const& src_vertex_weights_cache,
    edge_src_property_t<int64_t, int64_t> const& src_clusters_cache,
    edge_dst_property_t<int64_t, int64_t> const& dst_clusters_cache,
    rmm::device_uvector<int64_t>&& cluster_keys_v,
    rmm::device_uvector<float>&& cluster_weights_v,
    float total_edge_weight,
    float resolution,
    float theta);

template CUGRAPH_EXPORT
  std::tuple<vertex_property_t<int64_t, int64_t>,
             std::pair<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
  refine_clustering<int64_t, int64_t, true, double>(
    raft::handle_t const& handle,
    raft::random::RngState& rng_state,
    cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
    std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
    vertex_property_view_t<int64_t, double const*> vertex_weights_v,
    vertex_property_view_t<int64_t, int64_t const*> next_clusters_v,
    edge_src_property_t<int64_t, double> const& src_vertex_weights_cache,
    edge_src_property_t<int64_t, int64_t> const& src_clusters_cache,
    edge_dst_property_t<int64_t, int64_t> const& dst_clusters_cache,
    rmm::device_uvector<int64_t>&& cluster_keys_v,
    rmm::device_uvector<double>&& cluster_weights_v,
    double total_edge_weight,
    double resolution,
    double theta);

}  // namespace detail
}  // namespace cugraph

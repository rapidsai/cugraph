/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dendrogram.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/vertex_property.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu, typename weight_t>
std::tuple<vertex_property_t<vertex_t, vertex_t>,
           std::pair<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>>
refine_clustering(raft::handle_t const& handle,
                  raft::random::RngState& rng_state,
                  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                  vertex_property_view_t<vertex_t, weight_t const*> vertex_weights_v,
                  vertex_property_view_t<vertex_t, vertex_t const*> next_clusters_v,
                  edge_src_property_t<vertex_t, weight_t> const& src_vertex_weights_cache,
                  edge_src_property_t<vertex_t, vertex_t> const& src_clusters_cache,
                  edge_dst_property_t<vertex_t, vertex_t> const& dst_clusters_cache,
                  rmm::device_uvector<vertex_t>&& cluster_keys_v,
                  rmm::device_uvector<weight_t>&& cluster_weights_v,
                  weight_t total_edge_weight,
                  weight_t resolution,
                  weight_t theta);

}
}  // namespace cugraph

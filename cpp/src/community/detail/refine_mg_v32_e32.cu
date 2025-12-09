/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "community/detail/refine_impl.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>,
                    std::pair<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
refine_clustering(raft::handle_t const& handle,
                  raft::random::RngState& rng_state,
                  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                  float total_edge_weight,
                  float resolution,
                  float theta,
                  rmm::device_uvector<float> const& vertex_weights_v,
                  rmm::device_uvector<int32_t>&& cluster_keys_v,
                  rmm::device_uvector<float>&& cluster_weights_v,
                  rmm::device_uvector<int32_t>&& next_clusters_v,
                  edge_src_property_t<int32_t, float> const& src_vertex_weights_cache,
                  edge_src_property_t<int32_t, int32_t> const& src_clusters_cache,
                  edge_dst_property_t<int32_t, int32_t> const& dst_clusters_cache);

template std::tuple<rmm::device_uvector<int32_t>,
                    std::pair<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
refine_clustering(raft::handle_t const& handle,
                  raft::random::RngState& rng_state,
                  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                  double total_edge_weight,
                  double resolution,
                  double theta,
                  rmm::device_uvector<double> const& vertex_weights_v,
                  rmm::device_uvector<int32_t>&& cluster_keys_v,
                  rmm::device_uvector<double>&& cluster_weights_v,
                  rmm::device_uvector<int32_t>&& next_clusters_v,
                  edge_src_property_t<int32_t, double> const& src_vertex_weights_cache,
                  edge_src_property_t<int32_t, int32_t> const& src_clusters_cache,
                  edge_dst_property_t<int32_t, int32_t> const& dst_clusters_cache);

}  // namespace detail
}  // namespace cugraph

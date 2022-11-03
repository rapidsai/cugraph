/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <utilities/high_res_timer.hpp>

#include <cugraph/dendrogram.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename graph_view_t>
std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
           std::pair<rmm::device_uvector<typename graph_view_t::vertex_type>,
                     rmm::device_uvector<typename graph_view_t::vertex_type>>>
refine_clustering_2(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  typename graph_view_t::weight_type total_edge_weight,
  typename graph_view_t::weight_type resolution,
  rmm::device_uvector<typename graph_view_t::weight_type> const& vertex_weights_v,
  rmm::device_uvector<typename graph_view_t::vertex_type>&& cluster_keys_v,
  rmm::device_uvector<typename graph_view_t::weight_type>&& cluster_weights_v,
  rmm::device_uvector<typename graph_view_t::vertex_type>&& next_clusters_v,
  edge_src_property_t<graph_view_t, typename graph_view_t::weight_type> const&
    src_vertex_weights_cache,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache,
  edge_dst_property_t<graph_view_t, typename graph_view_t::vertex_type> const& dst_clusters_cache,
  bool up_down);

}
}  // namespace cugraph
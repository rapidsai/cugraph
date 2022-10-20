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

// Some timing functions
//   Need to #define TIMING to have these functions actually time, otherwise
//   this is a noop
template <bool multi_gpu>
void timer_start(raft::handle_t const& handle, HighResTimer& hr_timer, std::string const& region)
{
#ifdef TIMING
  if constexpr (multi_gpu) {
    if (handle.get_comms().get_rank() == 0) hr_timer.start(region);
  } else {
    hr_timer.start(region);
  }
#endif
}

template <bool multi_gpu>
void timer_stop(raft::handle_t const& handle, HighResTimer& hr_timer)
{
#ifdef TIMING
  if constexpr (multi_gpu) {
    if (handle.get_comms().get_rank() == 0) {
      handle.get_stream().synchronize();
      hr_timer.stop();
    }
  } else {
    handle.get_stream().synchronize();
    hr_timer.stop();
  }
#endif
}

template <bool multi_gpu>
void timer_display(raft::handle_t const& handle, HighResTimer const& hr_timer, std::ostream& os)
{
#ifdef TIMING
  if (multi_gpu) {
    if (handle.get_comms().get_rank() == 0) hr_timer.display(os);
  } else {
    hr_timer.display(os);
  }
#endif
}

template <typename graph_view_t>
typename graph_view_t::weight_type compute_modularity(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache,
  edge_dst_property_t<graph_view_t, typename graph_view_t::vertex_type> const& dst_clusters_cache,
  rmm::device_uvector<typename graph_view_t::vertex_type> const& next_clusters,
  rmm::device_uvector<typename graph_view_t::weight_type> const& cluster_weights,
  typename graph_view_t::weight_type total_edge_weight,
  typename graph_view_t::weight_type resolution);

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t> labels);

template <typename graph_view_t>
rmm::device_uvector<typename graph_view_t::vertex_type> update_clustering_by_delta_modularity(
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

template <typename graph_view_t>
std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
           std::pair<rmm::device_uvector<typename graph_view_t::vertex_type>,
                     rmm::device_uvector<typename graph_view_t::vertex_type>>>
refine_clustering(
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

template <typename graph_view_t>
std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
           rmm::device_uvector<typename graph_view_t::weight_type>>
compute_cluster_keys_and_values(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  rmm::device_uvector<typename graph_view_t::vertex_type> const& next_clusters_v,
  edge_src_property_t<graph_view_t, typename graph_view_t::vertex_type> const& src_clusters_cache);

}  // namespace detail
}  // namespace cugraph

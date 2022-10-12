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
//#define TIMING

#include <community/detail/common_methods.hpp>
#include <community/flatten_dendrogram.hpp>

// FIXME:  Only outstanding items preventing this becoming a .hpp file
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {

namespace detail {

// FIXME: Can we have a common check_clustering to be used by both
// Louvain and Leiden, and possibly other clustering methods?
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
void check_clustering(
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t* clustering)
{
  if (graph_view.local_vertex_partition_range_size() > 0)
    CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");
}

// template <typename vertex_t,
//           typename edge_t,
//           typename weight_t,
//           bool multi_gpu,
//           bool store_transposed = false>
// void move_node_fast(
//   raft::handle_t const& handle,
//   graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>& current_graph_view,
//   std::unique_ptr<Dendrogram<vertex_t>>& dendrogram)
// {
//   // Allocate queue to hold vertices assigned to this GPU
//   // vertex_frontier_t<vertex_t, void, multi_gpu> vertex_frontier(
//   //   handle_, graph_view.local_vertex_partition_range_size());
//   return;
// }

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
std::unique_ptr<rmm::device_uvector<vertex_t>> refine_partition(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  rmm::device_uvector<vertex_t>& partition,
  std::unique_ptr<Dendrogram<vertex_t>>& dendrogram)
{
  //
  // Every vertex starts as a singleton cluster in the refined partition
  //

  rmm::device_uvector<vertex_t> partition_r(current_graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        partition_r.begin(),
                        partition_r.size(),
                        current_graph_view.local_vertex_partition_range_first());

  // TODO: Refine incoming partition

  return std::move(refined_partition);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
void merge_nodes_subset(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  rmm::device_uvector<vertex_t>& partition,
  std::unique_ptr<Dendrogram<vertex_t>>& dendrogram)
{
  // Changes must be reflected in partition
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
void aggregate_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>& current_graph_view,
  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>& current_graph,
  std::unique_ptr<Dendrogram<vertex_t>>& dendrogram)
{
  current_graph = cugraph::detail::graph_contraction(
    handle,
    current_graph_view,
    raft::device_span<vertex_t>{dendrogram->current_level_begin(),
                                dendrogram->current_level_size()});
  current_graph_view = current_graph.view();
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> leiden(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  size_t max_level,
  weight_t resolution)
{
  using graph_t = cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>;
  using graph_view_t =
    cugraph::graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>;

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram = std::make_unique<Dendrogram<vertex_t>>();

  graph_t current_graph(handle);
  graph_view_t current_graph_view(graph_view);

  HighResTimer hr_timer;

  weight_t best_modularity   = weight_t{-1.0};
  weight_t total_edge_weight = current_graph_view.compute_total_edge_weight(handle);

  //
  // Bookkeeping per cluster
  //
  rmm::device_uvector<vertex_t> clusters_ids(0, handle.get_stream());     //#C
  rmm::device_uvector<weight_t> cluster_weights(0, handle.get_stream());  //#C

  //
  // Bookkeeping per vertex
  //
  rmm::device_uvector<weight_t> vertex_weights(0, handle.get_stream());                   //#V
  rmm::device_uvector<vertex_t> cluster_assignment_for_vertices(0, handle.get_stream());  //#V

  //
  // Edge source cache
  //
  edge_src_property_t<graph_view_t, weight_t> src_vertex_weights_cache(handle);
  edge_src_property_t<graph_view_t, vertex_t> src_vertex_cluster_assignment_cache(handle);
  edge_dst_property_t<graph_view_t, vertex_t> dst_vertex_cluster_assignment_cache(handle);

  while (dendrogram->num_levels() < max_level) {
    //
    //  Initialize every cluster to reference each vertex to itself
    //
    dendrogram->add_level(current_graph_view.local_vertex_partition_range_first(),
                          current_graph_view.local_vertex_partition_range_size(),
                          handle.get_stream());

    detail::sequence_fill(handle.get_stream(),
                          dendrogram->current_level_begin(),
                          dendrogram->current_level_size(),
                          current_graph_view.local_vertex_partition_range_first());

    //
    //  Compute the vertex and cluster weights, these are different for each
    //  graph in the hierarchical decomposition
    //
    detail::timer_start<graph_view_t::is_multi_gpu>(
      handle, hr_timer, "compute_vertex_and_cluster_weights");

    vertex_weights = current_graph_view.compute_out_weight_sums(handle);
    clusters_ids.resize(vertex_weights.size(), handle.get_stream());
    cluster_weights.resize(vertex_weights.size(), handle.get_stream());

    detail::sequence_fill(handle.get_stream(),
                          clusters_ids.begin(),
                          clusters_ids.size(),
                          current_graph_view.local_vertex_partition_range_first());

    raft::copy(
      cluster_weights.begin(), vertex_weights.begin(), vertex_weights.size(), handle.get_stream());

    if constexpr (graph_view_t::is_multi_gpu) {
      std::tie(clusters_ids, cluster_weights) = shuffle_ext_vertices_and_values_by_gpu_id(
        handle, std::move(clusters_ids), std::move(cluster_weights));

      //
      // Hash(cluster_id) % #GPUs => current GPU
      //

      src_vertex_weights_cache =
        edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view);
      update_edge_src_property(
        handle, current_graph_view, vertex_weights.begin(), src_vertex_weights_cache);
      vertex_weights.resize(0, handle.get_stream());
      vertex_weights.shrink_to_fit(handle.get_stream());
    }

    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);

    //
    //  Update the clustering assignment, this is the main loop of Louvain
    //
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "update_clustering");

    cluster_assignment_for_vertices =
      rmm::device_uvector<vertex_t>(dendrogram->current_level_size(), handle.get_stream());

    // rmm::device_uvector<uint8_t> candidate_flags(0, handle.get_stream()); //#V
    // candidate_flags = rmm::device_uvector<uint8_t>(current_graph_view.number_of_vertices(),
    // handle.get_stream()); thrust::fill(handle.get_thrust_policy(), candidate_flags.begin(),
    // candidate_flags.end(), uint8_t{1});

    raft::copy(cluster_assignment_for_vertices.begin(),
               dendrogram->current_level_begin(),
               dendrogram->current_level_size(),
               handle.get_stream());

    if constexpr (multi_gpu) {
      src_vertex_cluster_assignment_cache =
        edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_src_property(handle,
                               current_graph_view,
                               cluster_assignment_for_vertices.begin(),
                               src_vertex_cluster_assignment_cache);
      dst_vertex_cluster_assignment_cache =
        edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_dst_property(handle,
                               current_graph_view,
                               cluster_assignment_for_vertices.begin(),
                               dst_vertex_cluster_assignment_cache);

      // Couldn't we clear cluster_assignment_for_vertices here?
    }

    weight_t new_Q = detail::compute_modularity(handle,
                                                current_graph_view,
                                                src_vertex_cluster_assignment_cache,
                                                dst_vertex_cluster_assignment_cache,
                                                cluster_assignment_for_vertices,
                                                cluster_weights,
                                                total_edge_weight,
                                                resolution);
    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap clusters_ids
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;

    while (new_Q > (cur_Q + 0.0001)) {
      cur_Q = new_Q;

      //
      // Keep a copy of detail::update_clustering_by_delta_modularity if we want to
      // resue detail::update_clustering_by_delta_modularity without changing
      //

      //
      // FIX: Existing detail::update_clustering_by_delta_modularity is slow.
      // To make is faster as proposed by Leiden algorithm, 1) keep track of the
      // vertices that have moved. And then 2) for all the vertices that have moved,
      // check if their neighbors belong to the same community.
      // If the neighbors belong to different communities, the collect them in a queue/list
      // In the next iteration, only conside vertices in the queue/list, until there the
      // queue/list is empty.
      //
      // IMPORTANT NOTE: Need to think which vertices are considered first
      //

      cluster_assignment_for_vertices =
        detail::update_clustering_by_delta_modularity(handle,
                                                      current_graph_view,
                                                      total_edge_weight,
                                                      resolution,
                                                      vertex_weights,
                                                      std::move(clusters_ids),
                                                      std::move(cluster_weights),
                                                      std::move(cluster_assignment_for_vertices),
                                                      src_vertex_weights_cache,
                                                      src_vertex_cluster_assignment_cache,
                                                      dst_vertex_cluster_assignment_cache,
                                                      up_down);

      //
      // Count number of unique clusters (aka partitions) and if it's same as before then break
      //

      rmm::device_uvector<vertex_t> copied_partition(copied_partition.size(), handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   cluster_assignment_for_vertices.begin(),
                   cluster_assignment_for_vertices.end(),
                   copied_partition.begin());

      thrust::sort(handle.get_thrust_policy(), copied_partition.begin(), copied_partition.end());

      auto nr_unique_clusters = static_cast<vertex_t>(thrust::distance(
        copied_partition.begin(),
        thrust::unique(
          handle.get_thrust_policy(), copied_partition.begin(), copied_partition.end())));

      if constexpr (graph_view_t::is_multi_gpu) {
        copied_partition.resize(nr_unique_clusters);
        copied_partition =
          shuffle_ext_vertices_and_values_by_gpu_id(handle, std::move(copied_partition));

        thrust::sort(handle.get_thrust_policy(), copied_partition.begin(), copied_partition.end());

        nr_unique_clusters = static_cast<vertex_t>(thrust::distance(
          copied_partition.begin(),
          thrust::unique(
            handle.get_thrust_policy(), copied_partition.begin(), copied_partition.end())));

        nr_unique_clusters = host_scalar_allreduce(
          handle.get_comms(), nr_unique_clusters, raft::comms::op_t::SUM, handle.get_stream());
      }

      if (nr_unique_clusters == current_graph_view.number_of_vertices()) { break; }

      //
      // Refine the current partition
      //

      if constexpr (graph_view_t::is_multi_gpu) {
        update_edge_src_property(handle,
                                 current_graph_view,
                                 cluster_assignment_for_vertices.begin(),
                                 src_vertex_cluster_assignment_cache);
        update_edge_dst_property(handle,
                                 current_graph_view,
                                 cluster_assignment_for_vertices.begin(),
                                 dst_vertex_cluster_assignment_cache);
      }

      std::tie(clusters_ids, cluster_weights) =
        detail::compute_cluster_keys_and_values(handle,
                                                current_graph_view,
                                                cluster_assignment_for_vertices,
                                                src_vertex_cluster_assignment_cache);

      up_down = !up_down;

      new_Q = detail::compute_modularity(handle,
                                         current_graph_view,
                                         src_vertex_cluster_assignment_cache,
                                         dst_vertex_cluster_assignment_cache,
                                         cluster_assignment_for_vertices,
                                         cluster_weights,
                                         total_edge_weight,
                                         resolution);

      if (new_Q > cur_Q) {
        raft::copy(dendrogram->current_level_begin(),
                   cluster_assignment_for_vertices.begin(),
                   cluster_assignment_for_vertices.size(),
                   handle.get_stream());
      }
    }

    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);

    if (cur_Q <= best_modularity) { break; }

    best_modularity = cur_Q;

    //
    // Clear buffer and contract the graph
    //

    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "contract graph");

    clusters_ids.resize(0, handle.get_stream());
    cluster_weights.resize(0, handle.get_stream());
    vertex_weights.resize(0, handle.get_stream());
    cluster_assignment_for_vertices.resize(0, handle.get_stream());
    clusters_ids.shrink_to_fit(handle.get_stream());
    cluster_weights.shrink_to_fit(handle.get_stream());
    vertex_weights.shrink_to_fit(handle.get_stream());
    cluster_assignment_for_vertices.shrink_to_fit(handle.get_stream());
    src_vertex_weights_cache.clear(handle);
    src_vertex_cluster_assignment_cache.clear(handle);
    dst_vertex_cluster_assignment_cache.clear(handle);

    aggregate_graph(handle, current_graph_view, current_graph, dendrogram);

    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
  }

  detail::timer_display<graph_view_t::is_multi_gpu>(handle, hr_timer, std::cout);

  return std::make_pair(std::move(dendrogram), best_modularity);
  // TODO: everything
  // CUGRAPH_FAIL("unimplemented");
  // return std::make_pair(std::make_unique<Dendrogram<vertex_t>>(), weight_t{0.0});
}

// FIXME: Can we have a common flatten_dendrogram to be used by both
// Louvain and Leiden, and possibly other clustering methods?
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
void flatten_dendrogram(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  Dendrogram<vertex_t> const& dendrogram,
  vertex_t* clustering)
{
  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices(), handle.get_stream());

  thrust::sequence(handle.get_thrust_policy(),
                   vertex_ids_v.begin(),
                   vertex_ids_v.end(),
                   graph_view.local_vertex_partition_range_first());

  partition_at_level<vertex_t, multi_gpu>(
    handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
}

}  // namespace detail

template <typename GraphViewType>
std::pair<std::unique_ptr<Dendrogram<typename GraphViewType::vertex_type>>,
          typename GraphViewType::weight_type>
leiden(raft::handle_t const& handle,
       GraphViewType const& graph_view,
       size_t max_level,
       typename GraphViewType::weight_type resolution)
{
  return detail::leiden(handle, graph_view, max_level, resolution);
}

template <typename GraphViewType>
void flatten_dendrogram(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        Dendrogram<typename GraphViewType::vertex_type> const& dendrogram,
                        typename GraphViewType::vertex_type* clustering)
{
  detail::flatten_dendrogram(handle, graph_view, dendrogram, clustering);
}

}  // namespace cugraph

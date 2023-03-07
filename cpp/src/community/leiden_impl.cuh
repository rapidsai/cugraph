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
// #define TIMING

#include <community/detail/common_methods.hpp>
#include <community/detail/refine.hpp>
#include <community/flatten_dendrogram.hpp>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <rmm/device_uvector.hpp>

namespace cugraph {

namespace detail {

// FIXME: Can we have a common check_clustering to be used by both
// Louvain and Leiden, and possibly other clustering methods?
template <typename vertex_t, typename edge_t, bool multi_gpu>
void check_clustering(graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      vertex_t* clustering)
{
  if (graph_view.local_vertex_partition_range_size() > 0)
    CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> leiden(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> input_edge_weight_view,
  size_t max_level,
  weight_t resolution)
{
  using graph_t      = cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>;
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>;

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram = std::make_unique<Dendrogram<vertex_t>>();

  graph_t current_graph(handle);
  graph_view_t current_graph_view(graph_view);

  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view(
    input_edge_weight_view);
  std::optional<edge_property_t<graph_view_t, weight_t>> coarsen_graph_edge_property(handle);

#ifdef TIMING
  HighResTimer hr_timer{};
#endif

  weight_t best_modularity = weight_t{-1.0};
  weight_t total_edge_weight =
    compute_total_edge_weight(handle, current_graph_view, *edge_weight_view);

  //
  // Bookkeeping per cluster
  //
  rmm::device_uvector<vertex_t> cluster_keys(0, handle.get_stream());     // #C
  rmm::device_uvector<weight_t> cluster_weights(0, handle.get_stream());  // #C

  rmm::device_uvector<vertex_t> tmp_cluster_keys(0, handle.get_stream());     // #C
  rmm::device_uvector<weight_t> tmp_cluster_weights(0, handle.get_stream());  // #C

  rmm::device_uvector<vertex_t> tmp_cluster_keys(0, handle.get_stream());     //#C
  rmm::device_uvector<weight_t> tmp_cluster_weights(0, handle.get_stream());  //#C

  //
  // Bookkeeping per vertex
  //
  rmm::device_uvector<weight_t> vertex_weights(0, handle.get_stream());                   // #V
  rmm::device_uvector<vertex_t> louvain_assignment_for_vertices(0, handle.get_stream());  // #V
  rmm::device_uvector<vertex_t> louvain_of_refined_partition(0, handle.get_stream());     // #V

  //
  // Edge source cache
  //
  edge_src_property_t<graph_view_t, weight_t> src_vertex_weights_cache(handle);
  edge_src_property_t<graph_view_t, vertex_t> src_louvain_assignment_cache(handle);
  edge_dst_property_t<graph_view_t, vertex_t> dst_louvain_assignment_cache(handle);

  std::cout << "#V: " << current_graph_view.local_vertex_partition_range_size() << std::endl;
  std::cout << "#E: " << graph_view.local_edge_partition_view(0).number_of_edges() << std::endl;

  if (graph_view_t::is_multi_gpu) {
    std::cout << "Multi GPU graph" << std::endl;
  } else {
    std::cout << "Singple GPU graph" << std::endl;
  }

  if (multi_gpu) {
    std::cout << "multi_gpu = true" << std::endl;
  } else {
    std::cout << "multi_gpu = false" << std::endl;
  }

  bool first_iteration = true;
  while (dendrogram->num_levels() < max_level) {
    //
    //  Initialize every cluster to reference each vertex to itself
    //
    dendrogram->add_level(current_graph_view.local_vertex_partition_range_first(),
                          current_graph_view.local_vertex_partition_range_size(),
                          handle.get_stream());

    bool debug = current_graph_view.local_vertex_partition_range_size() < 50;

    auto offsets = current_graph_view.local_edge_partition_view(0).offsets();
    auto indices = current_graph_view.local_edge_partition_view(0).indices();

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());

      std::cout << "---- Graph -----: " << max_level << std::endl;
      raft::print_device_vector("offsets: ", offsets.data(), offsets.size(), std::cout);
      raft::print_device_vector("indices: ", indices.data(), indices.size(), std::cout);

      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "---------------- outer loop -------------------: " << max_level << std::endl;
      std::cout << "dendrogram->num_levels(): " << dendrogram->num_levels() << std::endl;
    }

//
//  Compute the vertex and cluster weights, these are different for each
//  graph in the hierarchical decomposition
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(
      handle, hr_timer, "compute_vertex_and_cluster_weights");
#endif

    vertex_weights = compute_out_weight_sums(handle, current_graph_view, *edge_weight_view);
    cluster_keys.resize(vertex_weights.size(), handle.get_stream());
    cluster_weights.resize(vertex_weights.size(), handle.get_stream());

    if (debug) {
      std::cout << "vertex_weights.size() : " << vertex_weights.size()
                << ", louvain_of_refined_partition.size(): " << louvain_of_refined_partition.size()
                << ", current_graph_view (#V): "
                << current_graph_view.local_vertex_partition_range_size() << std::endl;
    }

    if (first_iteration) {
      std::cout << "initialize dendrogram with sequence_fill" << std::endl;
      detail::sequence_fill(handle.get_stream(),
                            dendrogram->current_level_begin(),
                            dendrogram->current_level_size(),
                            current_graph_view.local_vertex_partition_range_first());

      detail::sequence_fill(handle.get_stream(),
                            cluster_keys.begin(),
                            cluster_keys.size(),
                            current_graph_view.local_vertex_partition_range_first());

      raft::copy(cluster_weights.begin(),
                 vertex_weights.begin(),
                 vertex_weights.size(),
                 handle.get_stream());

      if constexpr (graph_view_t::is_multi_gpu) {
        std::tie(cluster_keys, cluster_weights) =
          shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
            handle, std::move(cluster_keys), std::move(cluster_weights));
      }

      first_iteration = false;
    } else {
      std::cout << "initialize dendrogram with louvain_of_refined_partition" << std::endl;

      tmp_cluster_weights.resize(vertex_weights.size(), handle.get_stream());
      raft::copy(dendrogram->current_level_begin(),
                 louvain_of_refined_partition.begin(),
                 louvain_of_refined_partition.size(),
                 handle.get_stream());

      raft::copy(tmp_cluster_weights.begin(),
                 vertex_weights.begin(),
                 vertex_weights.size(),
                 handle.get_stream());

      thrust::sort_by_key(handle.get_thrust_policy(),
                          louvain_of_refined_partition.begin(),
                          louvain_of_refined_partition.end(),
                          tmp_cluster_weights.begin());

      auto pair_of_iterators_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                                         louvain_of_refined_partition.begin(),
                                                         louvain_of_refined_partition.end(),
                                                         tmp_cluster_weights.begin(),
                                                         cluster_keys.begin(),
                                                         cluster_weights.begin());

      cluster_keys.resize(
        static_cast<size_t>(thrust::distance(cluster_keys.begin(), pair_of_iterators_end.first)),
        handle.get_stream());
      cluster_weights.resize(static_cast<size_t>(thrust::distance(cluster_weights.begin(),
                                                                  pair_of_iterators_end.second)),
                             handle.get_stream());
      tmp_cluster_weights.resize(0, handle.get_stream());

      if constexpr (graph_view_t::is_multi_gpu) {
        std::tie(tmp_cluster_keys, tmp_cluster_weights) =
          shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
            handle, std::move(cluster_keys), std::move(cluster_weights));

        cluster_keys.resize(tmp_cluster_keys.size(), handle.get_stream());
        cluster_weights.resize(tmp_cluster_weights.size(), handle.get_stream());

        thrust::sort_by_key(handle.get_thrust_policy(),
                            tmp_cluster_keys.begin(),
                            tmp_cluster_keys.end(),
                            tmp_cluster_weights.begin());
        pair_of_iterators_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                                      tmp_cluster_keys.begin(),
                                                      tmp_cluster_keys.end(),
                                                      tmp_cluster_weights.begin(),
                                                      cluster_keys.begin(),
                                                      cluster_weights.begin());

        cluster_keys.resize(
          static_cast<size_t>(thrust::distance(cluster_keys.begin(), pair_of_iterators_end.first)),
          handle.get_stream());
        cluster_weights.resize(static_cast<size_t>(thrust::distance(cluster_weights.begin(),
                                                                    pair_of_iterators_end.second)),
                               handle.get_stream());
      }
    }

    if constexpr (graph_view_t::is_multi_gpu) {
      src_vertex_weights_cache =
        edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view);
      update_edge_src_property(
        handle, current_graph_view, vertex_weights.begin(), src_vertex_weights_cache);
      vertex_weights.resize(0, handle.get_stream());
      vertex_weights.shrink_to_fit(handle.get_stream());
    }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

//  Update the clustering assignment, this is the main loop of Louvain
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "update_clustering");
#endif

    louvain_assignment_for_vertices =
      rmm::device_uvector<vertex_t>(dendrogram->current_level_size(), handle.get_stream());

    raft::copy(louvain_assignment_for_vertices.begin(),
               dendrogram->current_level_begin(),
               dendrogram->current_level_size(),
               handle.get_stream());

    if constexpr (multi_gpu) {
      src_louvain_assignment_cache =
        edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_src_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.begin(),
                               src_louvain_assignment_cache);
      dst_louvain_assignment_cache =
        edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_dst_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.begin(),
                               dst_louvain_assignment_cache);

      // Couldn't we clear louvain_assignment_for_vertices here?
    }

    weight_t new_Q = detail::compute_modularity(handle,
                                                current_graph_view,
                                                edge_weight_view,
                                                src_louvain_assignment_cache,
                                                dst_louvain_assignment_cache,
                                                louvain_assignment_for_vertices,
                                                cluster_weights,
                                                total_edge_weight,
                                                resolution);
    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap cluster_keys
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down             = true;
    bool no_movement         = true;
    int32_t inner_loop_count = 0;
    while (new_Q > (cur_Q + 1e-4)) {
      cur_Q = new_Q;

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        std::cout << "------------ inner loop, counter: " << inner_loop_count++ << std::endl;
      }
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

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        std::cout << " total_edge_weight: " << total_edge_weight << std::endl;
        std::cout << " resolution: " << resolution << std::endl;

        raft::print_device_vector(
          "cluster_keys: ", cluster_keys.data(), cluster_keys.size(), std::cout);

        raft::print_device_vector(
          "cluster_weights: ", cluster_weights.data(), cluster_weights.size(), std::cout);

        std::cout << "Before update_clustering_by_delta_modularity ..." << std::endl;
        raft::print_device_vector("louvain_assignment_for_vertices: ",
                                  louvain_assignment_for_vertices.data(),
                                  louvain_assignment_for_vertices.size(),
                                  std::cout);
        raft::print_device_vector("*edge_weight_view: ",
                                  (*edge_weight_view).value_firsts()[0],
                                  std::min((*edge_weight_view).edge_counts()[0],
                                           (decltype((*edge_weight_view).edge_counts()[0]))50),
                                  std::cout);
      }

      louvain_assignment_for_vertices =
        detail::update_clustering_by_delta_modularity(handle,
                                                      current_graph_view,
                                                      edge_weight_view,
                                                      total_edge_weight,
                                                      resolution,
                                                      vertex_weights,
                                                      std::move(cluster_keys),
                                                      std::move(cluster_weights),
                                                      std::move(louvain_assignment_for_vertices),
                                                      src_vertex_weights_cache,
                                                      src_louvain_assignment_cache,
                                                      dst_louvain_assignment_cache,
                                                      up_down);
      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        std::cout << "After update_clustering_by_delta_modularity ..." << std::endl;
        raft::print_device_vector("*edge_weight_view: ",
                                  (*edge_weight_view).value_firsts()[0],
                                  std::min((*edge_weight_view).edge_counts()[0],
                                           (decltype((*edge_weight_view).edge_counts()[0]))50),
                                  std::cout);
        raft::print_device_vector("louvain_assignment_for_vertices: ",
                                  louvain_assignment_for_vertices.data(),
                                  louvain_assignment_for_vertices.size(),
                                  std::cout);
      }

      if constexpr (graph_view_t::is_multi_gpu) {
        update_edge_src_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.begin(),
                                 src_louvain_assignment_cache);
        update_edge_dst_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.begin(),
                                 dst_louvain_assignment_cache);
      }

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector("*edge_weight_view: ",
                                  (*edge_weight_view).value_firsts()[0],
                                  std::min((*edge_weight_view).edge_counts()[0],
                                           (decltype((*edge_weight_view).edge_counts()[0]))50),
                                  std::cout);
      }

      std::tie(cluster_keys, cluster_weights) =
        detail::compute_cluster_keys_and_values(handle,
                                                current_graph_view,
                                                edge_weight_view,
                                                louvain_assignment_for_vertices,
                                                src_louvain_assignment_cache);

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector("*edge_weight_view: ",
                                  (*edge_weight_view).value_firsts()[0],
                                  std::min((*edge_weight_view).edge_counts()[0],
                                           (decltype((*edge_weight_view).edge_counts()[0]))50),
                                  std::cout);
        raft::print_device_vector(
          "cluster_keys: ", cluster_keys.data(), cluster_keys.size(), std::cout);
        raft::print_device_vector(
          "cluster_weights: ", cluster_weights.data(), cluster_weights.size(), std::cout);
      }

      up_down = !up_down;

      new_Q = detail::compute_modularity(handle,
                                         current_graph_view,
                                         edge_weight_view,
                                         src_louvain_assignment_cache,
                                         dst_louvain_assignment_cache,
                                         louvain_assignment_for_vertices,
                                         cluster_weights,
                                         total_edge_weight,
                                         resolution);

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        std::cout << "new_Q: " << new_Q << std::endl;
        std::cout << "cur_Q: " << cur_Q << std::endl;
      }

      if (new_Q > (cur_Q + 1e-4)) {
        raft::copy(dendrogram->current_level_begin(),
                   louvain_assignment_for_vertices.begin(),
                   louvain_assignment_for_vertices.size(),
                   handle.get_stream());
        no_movement = false;
      }

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector("dendrogram: ",
                                  dendrogram->current_level_begin(),
                                  dendrogram->current_level_size(),
                                  std::cout);
      }

      if (debug) {
        CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector("dendrogram: ",
                                  dendrogram->current_level_begin(),
                                  dendrogram->current_level_size(),
                                  std::cout);
      }
    }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

    if (no_movement) { break; }
    best_modularity = cur_Q;

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Out of outer while loop,  best_modularity: " << best_modularity << std::endl;
    }

    //
    // Count number of unique clusters (aka partitions) and check if it's same as
    // number of vertices in the current graph
    //

    rmm::device_uvector<vertex_t> copied_louvain_partition(louvain_assignment_for_vertices.size(),
                                                           handle.get_stream());

    thrust::copy(handle.get_thrust_policy(),
                 louvain_assignment_for_vertices.begin(),
                 louvain_assignment_for_vertices.end(),
                 copied_louvain_partition.begin());

    thrust::sort(
      handle.get_thrust_policy(), copied_louvain_partition.begin(), copied_louvain_partition.end());

    auto nr_unique_clusters =
      static_cast<vertex_t>(thrust::distance(copied_louvain_partition.begin(),
                                             thrust::unique(handle.get_thrust_policy(),
                                                            copied_louvain_partition.begin(),
                                                            copied_louvain_partition.end())));

    copied_louvain_partition.resize(nr_unique_clusters, handle.get_stream());

    if constexpr (graph_view_t::is_multi_gpu) {
      copied_louvain_partition =
        cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
          handle, std::move(copied_louvain_partition));

      thrust::sort(handle.get_thrust_policy(),
                   copied_louvain_partition.begin(),
                   copied_louvain_partition.end());

      nr_unique_clusters =
        static_cast<vertex_t>(thrust::distance(copied_louvain_partition.begin(),
                                               thrust::unique(handle.get_thrust_policy(),
                                                              copied_louvain_partition.begin(),
                                                              copied_louvain_partition.end())));

      nr_unique_clusters = host_scalar_allreduce(
        handle.get_comms(), nr_unique_clusters, raft::comms::op_t::SUM, handle.get_stream());
    }

    std::cout << "nr_unique_clusters: " << nr_unique_clusters
              << ", current_graph_view.number_of_vertices(): "
              << current_graph_view.number_of_vertices() << std::endl;

    if (nr_unique_clusters == current_graph_view.number_of_vertices()) { break; }

    //
    // Refine the current partition
    //

    if constexpr (graph_view_t::is_multi_gpu) {
      update_edge_src_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.begin(),
                               src_louvain_assignment_cache);
      update_edge_dst_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.begin(),
                               dst_louvain_assignment_cache);
    }

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "refine_clustering .................." << std::endl;
    }
    auto [refined_leiden_partition, leiden_to_louvain_map] =
      detail::refine_clustering(handle,
                                current_graph_view,
                                edge_weight_view,
                                total_edge_weight,
                                resolution,
                                vertex_weights,
                                std::move(cluster_keys),
                                std::move(cluster_weights),
                                std::move(louvain_assignment_for_vertices),
                                src_vertex_weights_cache,
                                src_louvain_assignment_cache,
                                dst_louvain_assignment_cache,
                                up_down);

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());

      raft::print_device_vector("refined_leiden_partition: ",
                                refined_leiden_partition.data(),
                                refined_leiden_partition.size(),
                                std::cout);

      raft::print_device_vector("leiden_to_louvain_map (key): ",
                                leiden_to_louvain_map.first.data(),
                                leiden_to_louvain_map.first.size(),
                                std::cout);

      raft::print_device_vector("leiden_to_louvain_map (value): ",
                                leiden_to_louvain_map.second.data(),
                                leiden_to_louvain_map.second.size(),
                                std::cout);
    }

// Clear buffer and contract the graph
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "contract graph");
#endif

    cluster_keys.resize(0, handle.get_stream());
    cluster_weights.resize(0, handle.get_stream());
    vertex_weights.resize(0, handle.get_stream());
    louvain_assignment_for_vertices.resize(0, handle.get_stream());
    cluster_keys.shrink_to_fit(handle.get_stream());
    cluster_weights.shrink_to_fit(handle.get_stream());
    vertex_weights.shrink_to_fit(handle.get_stream());
    louvain_assignment_for_vertices.shrink_to_fit(handle.get_stream());
    src_vertex_weights_cache.clear(handle);
    src_louvain_assignment_cache.clear(handle);
    dst_louvain_assignment_cache.clear(handle);

    // Create aggregate graph based on refined (leiden) partition

    std::optional<rmm::device_uvector<vertex_t>> cluster_assignment{std::nullopt};

    std::tie(current_graph, coarsen_graph_edge_property, cluster_assignment) =
      cugraph::detail::graph_contraction(
        handle,
        current_graph_view,
        edge_weight_view,
        raft::device_span<vertex_t>{refined_leiden_partition.begin(),
                                    refined_leiden_partition.size()});
    current_graph_view = current_graph.view();

    edge_weight_view = std::make_optional<edge_property_view_t<edge_t, weight_t const*>>(
      (*coarsen_graph_edge_property).view());

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "... Leiden assignment of aggregated_graph" << std::endl;
      raft::print_device_vector("Leiden_assignment of aggregated_graph: ",
                                (*cluster_assignment).data(),
                                (*cluster_assignment).size(),
                                std::cout);
    }

    relabel<vertex_t, multi_gpu>(
      handle,
      std::make_tuple(static_cast<vertex_t const*>(leiden_to_louvain_map.first.begin()),
                      static_cast<vertex_t const*>(leiden_to_louvain_map.second.begin())),
      leiden_to_louvain_map.first.size(),
      (*cluster_assignment).data(),
      (*cluster_assignment).size(),
      false);

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "... Louvain assignment of aggregated_graph (using relabel) " << std::endl;
      raft::print_device_vector("Louvain_assignment of aggregated_graph: ",
                                (*cluster_assignment).data(),
                                (*cluster_assignment).size(),
                                std::cout);
    }
    // After call to relabel, cluster_assignment contains louvain partition of the aggregated graph
    louvain_of_refined_partition.resize(current_graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());

    raft::copy(louvain_of_refined_partition.begin(),
               (*cluster_assignment).begin(),
               (*cluster_assignment).size(),
               handle.get_stream());

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      raft::print_device_vector("louvain_of_refined_partition: ",
                                louvain_of_refined_partition.data(),
                                louvain_of_refined_partition.size(),
                                std::cout);
    }
#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif
  }

#ifdef TIMING
  detail::timer_display<graph_view_t::is_multi_gpu>(handle, hr_timer, std::cout);
#endif

  return std::make_pair(std::move(dendrogram), best_modularity);
}

// FIXME: Can we have a common flatten_dendrogram to be used by both
// Louvain and Leiden, and possibly other clustering methods?
template <typename vertex_t, typename edge_t, bool multi_gpu>
void flatten_dendrogram(raft::handle_t const& handle,
                        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
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

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> leiden(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  size_t max_level,
  weight_t resolution)
{
  return detail::leiden(handle, graph_view, edge_weight_view, max_level, resolution);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void flatten_dendrogram(raft::handle_t const& handle,
                        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                        Dendrogram<vertex_t> const& dendrogram,
                        vertex_t* clustering)
{
  detail::flatten_dendrogram(handle, graph_view, dendrogram, clustering);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<size_t, weight_t> leiden(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_t* clustering,
  size_t max_level,
  weight_t resolution)
{
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  detail::check_clustering(graph_view, clustering);

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram;
  weight_t modularity;

  std::tie(dendrogram, modularity) =
    detail::leiden(handle, graph_view, edge_weight_view, max_level, resolution);

  detail::flatten_dendrogram(handle, graph_view, *dendrogram, clustering);

  return std::make_pair(dendrogram->num_levels(), modularity);
}

}  // namespace cugraph

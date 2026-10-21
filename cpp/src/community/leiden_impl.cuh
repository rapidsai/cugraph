/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "community/detail/common_methods.hpp"
#include "community/detail/refine.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/utilities/thrust_wrappers/unique.hpp>
#include <cugraph/vertex_property.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

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
std::pair<dendrogram_t<vertex_t>, weight_t> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  size_t max_level,
  weight_t resolution,
  weight_t theta = 1.0)
{
  using graph_t      = cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>;
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>;

  dendrogram_t<vertex_t> dendrogram{};

  graph_view_t current_graph_view(graph_view);
  std::optional<edge_property_view_t<edge_t, weight_t const*>> current_edge_weight_view(
    edge_weight_view);

  graph_t coarse_graph(handle);
  std::optional<edge_property_t<edge_t, weight_t>> coarsen_graph_edge_weight(handle);

#ifdef TIMING
  HighResTimer hr_timer{};
#endif

  weight_t final_Q{-1};

  weight_t total_edge_weight =
    compute_total_edge_weight(handle, current_graph_view, *current_edge_weight_view);

  rmm::device_uvector<vertex_t> louvain_of_refined_graph(0, handle.get_stream());  // #V

  while (dendrogram.num_levels() < max_level) {
    //
    //  Initialize every cluster to reference each vertex to itself
    //

    auto level_range_first = current_graph_view.local_vertex_partition_range_first();
    vertex_property_t<vertex_t, vertex_t> level_clusters(handle, current_graph_view);

//
//  Compute the vertex and cluster weights, these are different for each
//  graph in the hierarchical decomposition
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(
      handle, hr_timer, "compute_vertex_and_cluster_weights");
#endif

    vertex_property_t<vertex_t, weight_t> vertex_weights(
      compute_out_weight_sums(handle, current_graph_view, *current_edge_weight_view),
      current_graph_view.local_vertex_partition_range_size());
    rmm::device_uvector<vertex_t> cluster_keys(0, handle.get_stream());
    rmm::device_uvector<weight_t> cluster_weights(0, handle.get_stream());

    if (dendrogram.num_levels() == 0) {
      cluster_keys.resize(current_graph_view.local_vertex_partition_range_size(),
                          handle.get_stream());
      cluster_weights.resize(current_graph_view.local_vertex_partition_range_size(),
                             handle.get_stream());

      copy_v(handle,
             current_graph_view,
             vertex_property_view_t{thrust::make_counting_iterator(level_range_first),
                                    current_graph_view.local_vertex_partition_range_size()},
             level_clusters.mutable_view());

      cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                        cluster_keys.begin(),
                        cluster_keys.begin() + cluster_keys.size(),
                        current_graph_view.local_vertex_partition_range_first());

      copy_v(handle,
             current_graph_view,
             vertex_weights.view(),
             vertex_property_view_t{cluster_weights.begin(),
                                    current_graph_view.local_vertex_partition_range_size()});

      if constexpr (graph_view_t::is_multi_gpu) {
        std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
        vertex_properties.push_back(std::move(cluster_weights));
        std::tie(cluster_keys, vertex_properties) =
          shuffle_ext_vertices(handle, std::move(cluster_keys), std::move(vertex_properties));
        cluster_weights = std::move(std::get<rmm::device_uvector<weight_t>>(vertex_properties[0]));
      }
    } else {
      rmm::device_uvector<weight_t> tmp_weights_buffer(
        current_graph_view.local_vertex_partition_range_size(),
        handle.get_stream());  // #C

      copy_v(
        handle,
        current_graph_view,
        vertex_property_view_t<vertex_t, vertex_t const*>{
          louvain_of_refined_graph.data(), current_graph_view.local_vertex_partition_range_size()},
        level_clusters.mutable_view());

      copy_v(handle,
             current_graph_view,
             vertex_weights.view(),
             vertex_property_view_t{tmp_weights_buffer.begin(),
                                    current_graph_view.local_vertex_partition_range_size()});

      thrust::sort_by_key(handle.get_thrust_policy(),
                          louvain_of_refined_graph.begin(),
                          louvain_of_refined_graph.end(),
                          tmp_weights_buffer.begin());

      auto num_unique_louvain_clusters_in_refined_partition =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(louvain_of_refined_graph.size()),
                         is_first_in_run_t<vertex_t const*>{louvain_of_refined_graph.data()});

      cluster_keys.resize(num_unique_louvain_clusters_in_refined_partition, handle.get_stream());
      cluster_weights.resize(num_unique_louvain_clusters_in_refined_partition, handle.get_stream());

      thrust::reduce_by_key(handle.get_thrust_policy(),
                            louvain_of_refined_graph.begin(),
                            louvain_of_refined_graph.end(),
                            tmp_weights_buffer.begin(),
                            cluster_keys.begin(),
                            cluster_weights.begin());

      louvain_of_refined_graph.resize(0, handle.get_stream());
      louvain_of_refined_graph.shrink_to_fit(handle.get_stream());

      tmp_weights_buffer.resize(0, handle.get_stream());
      tmp_weights_buffer.shrink_to_fit(handle.get_stream());

      if constexpr (graph_view_t::is_multi_gpu) {
        rmm::device_uvector<vertex_t> tmp_keys_buffer(0, handle.get_stream());  // #C

        {
          std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
          vertex_properties.push_back(std::move(cluster_weights));
          std::tie(tmp_keys_buffer, vertex_properties) =
            shuffle_ext_vertices(handle, std::move(cluster_keys), std::move(vertex_properties));
          tmp_weights_buffer =
            std::move(std::get<rmm::device_uvector<weight_t>>(vertex_properties[0]));
        }

        thrust::sort_by_key(handle.get_thrust_policy(),
                            tmp_keys_buffer.begin(),
                            tmp_keys_buffer.end(),
                            tmp_weights_buffer.begin());

        num_unique_louvain_clusters_in_refined_partition =
          thrust::count_if(handle.get_thrust_policy(),
                           thrust::make_counting_iterator(size_t{0}),
                           thrust::make_counting_iterator(tmp_keys_buffer.size()),
                           is_first_in_run_t<vertex_t const*>{tmp_keys_buffer.data()});

        cluster_keys = rmm::device_uvector<vertex_t>(
          num_unique_louvain_clusters_in_refined_partition, handle.get_stream());
        cluster_weights = rmm::device_uvector<weight_t>(
          num_unique_louvain_clusters_in_refined_partition, handle.get_stream());

        thrust::reduce_by_key(handle.get_thrust_policy(),
                              tmp_keys_buffer.begin(),
                              tmp_keys_buffer.end(),
                              tmp_weights_buffer.begin(),
                              cluster_keys.begin(),
                              cluster_weights.begin());

        tmp_keys_buffer.resize(0, handle.get_stream());
        tmp_keys_buffer.shrink_to_fit(handle.get_stream());
        tmp_weights_buffer.resize(0, handle.get_stream());
        tmp_weights_buffer.shrink_to_fit(handle.get_stream());
      }
    }

    edge_src_property_t<vertex_t, weight_t> src_vertex_weights_cache(handle);
    if constexpr (graph_view_t::is_multi_gpu) {
      src_vertex_weights_cache =
        edge_src_property_t<vertex_t, weight_t>(handle, current_graph_view);
      update_edge_src_property(
        handle, current_graph_view, vertex_weights.view(), src_vertex_weights_cache.mutable_view());
    }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

//  Update the clustering assignment, this is the main loop of Louvain
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "update_clustering");
#endif

    vertex_property_t<vertex_t, vertex_t> louvain_assignment_for_vertices(handle,
                                                                          current_graph_view);
    copy_v(handle,
           current_graph_view,
           level_clusters.view(),
           louvain_assignment_for_vertices.mutable_view());

    edge_src_property_t<vertex_t, vertex_t> src_louvain_assignment_cache(handle);
    edge_dst_property_t<vertex_t, vertex_t> dst_louvain_assignment_cache(handle);
    if constexpr (multi_gpu) {
      src_louvain_assignment_cache =
        edge_src_property_t<vertex_t, vertex_t>(handle, current_graph_view);
      update_edge_src_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.view(),
                               src_louvain_assignment_cache.mutable_view());
      dst_louvain_assignment_cache =
        edge_dst_property_t<vertex_t, vertex_t>(handle, current_graph_view);
      update_edge_dst_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.view(),
                               dst_louvain_assignment_cache.mutable_view());
    }

    weight_t new_Q = detail::compute_modularity(handle,
                                                current_graph_view,
                                                current_edge_weight_view,
                                                src_louvain_assignment_cache,
                                                dst_louvain_assignment_cache,
                                                louvain_assignment_for_vertices.view(),
                                                cluster_weights,
                                                total_edge_weight,
                                                resolution);
    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap cluster_keys
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;
    while (new_Q > (cur_Q + 1e-4)) {
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

      louvain_assignment_for_vertices =
        detail::update_clustering_by_delta_modularity(handle,
                                                      current_graph_view,
                                                      current_edge_weight_view,
                                                      total_edge_weight,
                                                      resolution,
                                                      vertex_weights.view(),
                                                      std::move(cluster_keys),
                                                      std::move(cluster_weights),
                                                      std::move(louvain_assignment_for_vertices),
                                                      src_vertex_weights_cache,
                                                      src_louvain_assignment_cache,
                                                      dst_louvain_assignment_cache,
                                                      up_down,
                                                      weight_t{0});

      if constexpr (graph_view_t::is_multi_gpu) {
        update_edge_src_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.view(),
                                 src_louvain_assignment_cache.mutable_view());
        update_edge_dst_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.view(),
                                 dst_louvain_assignment_cache.mutable_view());
      }

      std::tie(cluster_keys, cluster_weights) =
        detail::compute_cluster_keys_and_values(handle,
                                                current_graph_view,
                                                current_edge_weight_view,
                                                louvain_assignment_for_vertices.view(),
                                                src_louvain_assignment_cache);

      up_down = !up_down;

      new_Q = detail::compute_modularity(handle,
                                         current_graph_view,
                                         current_edge_weight_view,
                                         src_louvain_assignment_cache,
                                         dst_louvain_assignment_cache,
                                         louvain_assignment_for_vertices.view(),
                                         cluster_weights,
                                         total_edge_weight,
                                         resolution);

      if (new_Q > (cur_Q + 1e-4)) {
        copy_v(handle,
               current_graph_view,
               louvain_assignment_for_vertices.view(),
               level_clusters.mutable_view());
      }
    }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "contract graph");
#endif

    // Count number of unique louvain clusters

    rmm::device_uvector<vertex_t> copied_louvain_partition(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
    copy_v(
      handle,
      current_graph_view,
      level_clusters.view(),
      vertex_property_view_t<vertex_t, vertex_t*>{
        copied_louvain_partition.data(), static_cast<vertex_t>(copied_louvain_partition.size())});
    auto nr_unique_louvain_clusters =
      remove_duplicates<vertex_t, multi_gpu>(handle, copied_louvain_partition);

    bool terminate = (nr_unique_louvain_clusters == current_graph_view.number_of_vertices());

    vertex_property_t<vertex_t, vertex_t> refined_leiden_partition(handle);
    std::pair<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> leiden_to_louvain_map{
      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
      rmm::device_uvector<vertex_t>(0, handle.get_stream())};

    if (!terminate) {
      // Refine the current partition
      copy_v(handle,
             current_graph_view,
             level_clusters.view(),
             louvain_assignment_for_vertices.mutable_view());

      if constexpr (graph_view_t::is_multi_gpu) {
        update_edge_src_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.view(),
                                 src_louvain_assignment_cache.mutable_view());
        update_edge_dst_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.view(),
                                 dst_louvain_assignment_cache.mutable_view());
      }

      std::tie(refined_leiden_partition, leiden_to_louvain_map) =
        detail::refine_clustering(handle,
                                  rng_state,
                                  current_graph_view,
                                  current_edge_weight_view,
                                  vertex_weights.view(),
                                  louvain_assignment_for_vertices.view(),
                                  src_vertex_weights_cache,
                                  src_louvain_assignment_cache,
                                  dst_louvain_assignment_cache,
                                  std::move(cluster_keys),
                                  std::move(cluster_weights),
                                  total_edge_weight,
                                  resolution,
                                  theta);
    }

    // Clear buffer and contract the graph
    final_Q = detail::compute_modularity(handle,
                                         current_graph_view,
                                         current_edge_weight_view,
                                         src_louvain_assignment_cache,
                                         dst_louvain_assignment_cache,
                                         louvain_assignment_for_vertices.view(),
                                         cluster_weights,
                                         total_edge_weight,
                                         resolution);

    cluster_keys.resize(0, handle.get_stream());
    cluster_weights.resize(0, handle.get_stream());
    cluster_keys.shrink_to_fit(handle.get_stream());
    cluster_weights.shrink_to_fit(handle.get_stream());
    vertex_weights                  = vertex_property_t<vertex_t, weight_t>(handle);
    louvain_assignment_for_vertices = vertex_property_t<vertex_t, vertex_t>(handle);
    src_vertex_weights_cache.clear();
    src_louvain_assignment_cache.clear();
    dst_louvain_assignment_cache.clear();

    if (!terminate) {
      src_louvain_assignment_cache.clear();
      dst_louvain_assignment_cache.clear();

      auto nr_unique_leiden = static_cast<vertex_t>(leiden_to_louvain_map.first.size());
      if (graph_view_t::is_multi_gpu) {
        nr_unique_leiden = host_scalar_allreduce(
          handle.get_comms(), nr_unique_leiden, raft::comms::op_t::SUM, handle.get_stream());
      }

      terminate = terminate || (nr_unique_leiden == current_graph_view.number_of_vertices());

      if (nr_unique_leiden < current_graph_view.number_of_vertices()) {
        // New approach, mimic Louvain, we'll store the Leiden results in the dendrogram
        copy_v(handle,
               current_graph_view,
               refined_leiden_partition.view(),
               level_clusters.mutable_view());

        std::vector<vertex_t> vertex_partition_range_lasts(
          current_graph_view.vertex_partition_range_lasts().begin(),
          current_graph_view.vertex_partition_range_lasts().end());

        // Create aggregate graph based on refined (leiden) partition
        std::optional<vertex_property_t<vertex_t, vertex_t>> coarse_renumber_map{std::nullopt};
        std::tie(coarse_graph, coarsen_graph_edge_weight, coarse_renumber_map) =
          coarsen_graph(handle,
                        current_graph_view,
                        current_edge_weight_view,
                        refined_leiden_partition.view(),
                        true);
        level_clusters = vertex_property_t<vertex_t, vertex_t>(
          renumber_ext_vertices(handle,
                                coarse_graph.view(),
                                level_clusters.release_buffer(),
                                coarse_renumber_map->view()),
          current_graph_view.local_vertex_partition_range_size());
        auto numbering_map = std::make_optional(coarse_renumber_map->release_buffer());

        current_graph_view = coarse_graph.view();

        current_edge_weight_view =
          std::make_optional<edge_property_view_t<edge_t, weight_t const*>>(
            (*coarsen_graph_edge_weight).view());

        louvain_of_refined_graph.resize(current_graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());
        rmm::device_uvector<vertex_t> numeric_sequence(
          current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

        cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                          numeric_sequence.data(),
                          numeric_sequence.data() + numeric_sequence.size(),
                          current_graph_view.local_vertex_partition_range_first());

        raft::copy(louvain_of_refined_graph.begin(),
                   numbering_map->data(),
                   numbering_map->size(),
                   handle.get_stream());

        louvain_of_refined_graph = relabel<vertex_t, multi_gpu>(
          handle,
          raft::device_span<vertex_t const>(leiden_to_louvain_map.first.data(),
                                            leiden_to_louvain_map.first.size()),
          raft::device_span<vertex_t const>(leiden_to_louvain_map.second.data(),
                                            leiden_to_louvain_map.second.size()),
          std::move(louvain_of_refined_graph),
          std::make_optional(raft::host_span<vertex_t const>(vertex_partition_range_lasts.data(),
                                                             vertex_partition_range_lasts.size())),
          false);

        // Relabel clusters so that each cluster is identified by the lowest vertex id
        // that is assigned to it.  Note that numbering_map and numeric_sequence go out
        // of scope at the end of this block, we will reuse their memory
        raft::copy(numbering_map->begin(),
                   louvain_of_refined_graph.data(),
                   louvain_of_refined_graph.size(),
                   handle.get_stream());

        cugraph::sort(handle.get_thrust_policy(),
                      thrust::make_zip_iterator(numbering_map->begin(), numeric_sequence.begin()),
                      thrust::make_zip_iterator(numbering_map->end(), numeric_sequence.end()));

        size_t new_size = cuda::std::distance(numbering_map->begin(),
                                              thrust::unique_by_key(handle.get_thrust_policy(),
                                                                    numbering_map->begin(),
                                                                    numbering_map->end(),
                                                                    numeric_sequence.begin())
                                                .first);

        numbering_map->resize(new_size, handle.get_stream());
        numeric_sequence.resize(new_size, handle.get_stream());

        if constexpr (multi_gpu) {
          {
            std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
            vertex_properties.push_back(std::move(numeric_sequence));
            std::tie(*numbering_map, vertex_properties) =
              shuffle_ext_vertices(handle, std::move(*numbering_map), std::move(vertex_properties));
            numeric_sequence =
              std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
          }

          cugraph::sort(handle.get_thrust_policy(),
                        thrust::make_zip_iterator(numbering_map->begin(), numeric_sequence.begin()),
                        thrust::make_zip_iterator(numbering_map->end(), numeric_sequence.end()));

          size_t new_size = cuda::std::distance(numbering_map->begin(),
                                                thrust::unique_by_key(handle.get_thrust_policy(),
                                                                      numbering_map->begin(),
                                                                      numbering_map->end(),
                                                                      numeric_sequence.begin())
                                                  .first);

          numbering_map->resize(new_size, handle.get_stream());
          numeric_sequence.resize(new_size, handle.get_stream());
        }

        louvain_of_refined_graph = relabel<vertex_t, multi_gpu>(
          handle,
          raft::device_span<vertex_t const>(numbering_map->data(), numbering_map->size()),
          raft::device_span<vertex_t const>(numeric_sequence.data(), numeric_sequence.size()),
          std::move(louvain_of_refined_graph),
          std::nullopt,
          false);
      }
    } else {
      // Reset dendrogram.
      //    FIXME: Revisit how dendrogram is populated
      copy_v(handle,
             current_graph_view,
             vertex_property_view_t{thrust::make_counting_iterator(level_range_first),
                                    current_graph_view.local_vertex_partition_range_size()},
             level_clusters.mutable_view());
    }

    copied_louvain_partition.resize(0, handle.get_stream());
    copied_louvain_partition.shrink_to_fit(handle.get_stream());

    dendrogram.add_level(std::move(level_clusters), level_range_first);

    if (terminate) { break; }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif
  }  // end of outer while

#ifdef TIMING
  detail::timer_display<graph_view_t::is_multi_gpu>(handle, hr_timer, std::cout);
#endif

  return std::make_pair(std::move(dendrogram), final_Q);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<dendrogram_t<vertex_t>, weight_t> leiden_dendrogram(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  size_t max_level,
  weight_t resolution,
  weight_t theta)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::leiden(
    handle, rng_state, graph_view, edge_weight_view, max_level, resolution, theta);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<std::unique_ptr<dendrogram_t<vertex_t>>, weight_t> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  size_t max_level,
  weight_t resolution,
  weight_t theta)
{
  auto [dendrogram, modularity] = cugraph::leiden_dendrogram(
    handle, rng_state, graph_view, edge_weight_view, max_level, resolution, theta);

  return std::make_pair(std::make_unique<dendrogram_t<vertex_t>>(std::move(dendrogram)),
                        modularity);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, vertex_t>, size_t, weight_t> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  bool compact_cluster_ids,
  size_t max_level,
  weight_t resolution,
  weight_t theta)
{
  auto [dendrogram, modularity] = cugraph::leiden_dendrogram(
    handle, rng_state, graph_view, edge_weight_view, max_level, resolution, theta);
  auto clustering =
    cugraph::flatten_dendrogram(handle, graph_view, dendrogram, compact_cluster_ids);

  return std::make_tuple(std::move(clustering), dendrogram.num_levels(), modularity);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<size_t, weight_t> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_t* clustering,
  size_t max_level,
  weight_t resolution,
  weight_t theta)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  detail::check_clustering(graph_view, clustering);

  auto [clustering_prop, num_levels, modularity] = cugraph::leiden(
    handle, rng_state, graph_view, edge_weight_view, true, max_level, resolution, theta);
  copy_v(handle,
         graph_view,
         clustering_prop.view(),
         vertex_property_view_t<vertex_t, vertex_t*, vertex_t>{
           clustering, graph_view.local_vertex_partition_range_size()});

  return std::make_pair(num_levels, modularity);
}

}  // namespace cugraph

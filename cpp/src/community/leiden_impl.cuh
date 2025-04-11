/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "community/detail/common_methods.hpp"
#include "community/detail/refine.hpp"
#include "community/flatten_dendrogram.hpp"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
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
template <typename vertex_t, bool multi_gpu>
vertex_t remove_duplicates(raft::handle_t const& handle, rmm::device_uvector<vertex_t>& input_array)
{
  thrust::sort(handle.get_thrust_policy(), input_array.begin(), input_array.end());

  auto nr_unique_elements = static_cast<vertex_t>(cuda::std::distance(
    input_array.begin(),
    thrust::unique(handle.get_thrust_policy(), input_array.begin(), input_array.end())));

  input_array.resize(nr_unique_elements, handle.get_stream());

  if constexpr (multi_gpu) {
    input_array = shuffle_ext_vertices(handle, std::move(input_array));

    thrust::sort(handle.get_thrust_policy(), input_array.begin(), input_array.end());

    nr_unique_elements = static_cast<vertex_t>(cuda::std::distance(
      input_array.begin(),
      thrust::unique(handle.get_thrust_policy(), input_array.begin(), input_array.end())));

    input_array.resize(nr_unique_elements, handle.get_stream());

    nr_unique_elements = host_scalar_allreduce(
      handle.get_comms(), nr_unique_elements, raft::comms::op_t::SUM, handle.get_stream());
  }
  return nr_unique_elements;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed = false>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> leiden(
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

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram = std::make_unique<Dendrogram<vertex_t>>();

  graph_view_t current_graph_view(graph_view);
  std::optional<edge_property_view_t<edge_t, weight_t const*>> current_edge_weight_view(
    edge_weight_view);

  graph_t coarse_graph(handle);
  std::optional<edge_property_t<graph_view_t, weight_t>> coarsen_graph_edge_weight(handle);

#ifdef TIMING
  HighResTimer hr_timer{};
#endif

  weight_t final_Q{-1};

  weight_t total_edge_weight =
    compute_total_edge_weight(handle, current_graph_view, *current_edge_weight_view);

  rmm::device_uvector<vertex_t> louvain_of_refined_graph(0, handle.get_stream());  // #V

  while (dendrogram->num_levels() < max_level) {
    //
    //  Initialize every cluster to reference each vertex to itself
    //

    dendrogram->add_level(current_graph_view.local_vertex_partition_range_first(),
                          current_graph_view.local_vertex_partition_range_size(),
                          handle.get_stream());

//
//  Compute the vertex and cluster weights, these are different for each
//  graph in the hierarchical decomposition
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(
      handle, hr_timer, "compute_vertex_and_cluster_weights");
#endif

    rmm::device_uvector<weight_t> vertex_weights =
      compute_out_weight_sums(handle, current_graph_view, *current_edge_weight_view);
    rmm::device_uvector<vertex_t> cluster_keys(0, handle.get_stream());
    rmm::device_uvector<weight_t> cluster_weights(0, handle.get_stream());

    if (dendrogram->num_levels() == 1) {
      cluster_keys.resize(vertex_weights.size(), handle.get_stream());
      cluster_weights.resize(vertex_weights.size(), handle.get_stream());

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
        std::tie(cluster_keys, cluster_weights) = shuffle_ext_vertex_value_pairs(
          handle, std::move(cluster_keys), std::move(cluster_weights));
      }

    } else {
      rmm::device_uvector<weight_t> tmp_weights_buffer(vertex_weights.size(),
                                                       handle.get_stream());  // #C

      raft::copy(dendrogram->current_level_begin(),
                 louvain_of_refined_graph.begin(),
                 louvain_of_refined_graph.size(),
                 handle.get_stream());

      raft::copy(tmp_weights_buffer.begin(),
                 vertex_weights.begin(),
                 vertex_weights.size(),
                 handle.get_stream());

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

        std::tie(tmp_keys_buffer, tmp_weights_buffer) = shuffle_ext_vertex_value_pairs(
          handle, std::move(cluster_keys), std::move(cluster_weights));

        thrust::sort_by_key(handle.get_thrust_policy(),
                            tmp_keys_buffer.begin(),
                            tmp_keys_buffer.end(),
                            tmp_weights_buffer.begin());

        num_unique_louvain_clusters_in_refined_partition =
          thrust::count_if(handle.get_thrust_policy(),
                           thrust::make_counting_iterator(size_t{0}),
                           thrust::make_counting_iterator(tmp_keys_buffer.size()),
                           is_first_in_run_t<vertex_t const*>{tmp_keys_buffer.data()});

        cluster_keys.resize(num_unique_louvain_clusters_in_refined_partition, handle.get_stream());
        cluster_weights.resize(num_unique_louvain_clusters_in_refined_partition,
                               handle.get_stream());

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

    edge_src_property_t<graph_view_t, weight_t> src_vertex_weights_cache(handle);
    if constexpr (graph_view_t::is_multi_gpu) {
      src_vertex_weights_cache =
        edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view);
      update_edge_src_property(handle,
                               current_graph_view,
                               vertex_weights.begin(),
                               src_vertex_weights_cache.mutable_view());
    }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

//  Update the clustering assignment, this is the main loop of Louvain
#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "update_clustering");
#endif

    rmm::device_uvector<vertex_t> louvain_assignment_for_vertices(dendrogram->current_level_size(),
                                                                  handle.get_stream());

    raft::copy(louvain_assignment_for_vertices.begin(),
               dendrogram->current_level_begin(),
               dendrogram->current_level_size(),
               handle.get_stream());

    edge_src_property_t<graph_view_t, vertex_t> src_louvain_assignment_cache(handle);
    edge_dst_property_t<graph_view_t, vertex_t> dst_louvain_assignment_cache(handle);
    if constexpr (multi_gpu) {
      src_louvain_assignment_cache =
        edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_src_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.begin(),
                               src_louvain_assignment_cache.mutable_view());
      dst_louvain_assignment_cache =
        edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      update_edge_dst_property(handle,
                               current_graph_view,
                               louvain_assignment_for_vertices.begin(),
                               dst_louvain_assignment_cache.mutable_view());
    }

    weight_t new_Q = detail::compute_modularity(handle,
                                                current_graph_view,
                                                current_edge_weight_view,
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
                                                      vertex_weights,
                                                      std::move(cluster_keys),
                                                      std::move(cluster_weights),
                                                      std::move(louvain_assignment_for_vertices),
                                                      src_vertex_weights_cache,
                                                      src_louvain_assignment_cache,
                                                      dst_louvain_assignment_cache,
                                                      up_down);

      if constexpr (graph_view_t::is_multi_gpu) {
        update_edge_src_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.begin(),
                                 src_louvain_assignment_cache.mutable_view());
        update_edge_dst_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.begin(),
                                 dst_louvain_assignment_cache.mutable_view());
      }

      std::tie(cluster_keys, cluster_weights) =
        detail::compute_cluster_keys_and_values(handle,
                                                current_graph_view,
                                                current_edge_weight_view,
                                                louvain_assignment_for_vertices,
                                                src_louvain_assignment_cache);

      up_down = !up_down;

      new_Q = detail::compute_modularity(handle,
                                         current_graph_view,
                                         current_edge_weight_view,
                                         src_louvain_assignment_cache,
                                         dst_louvain_assignment_cache,
                                         louvain_assignment_for_vertices,
                                         cluster_weights,
                                         total_edge_weight,
                                         resolution);

      if (new_Q > (cur_Q + 1e-4)) {
        raft::copy(dendrogram->current_level_begin(),
                   louvain_assignment_for_vertices.begin(),
                   louvain_assignment_for_vertices.size(),
                   handle.get_stream());
      }
    }

#ifdef TIMING
    detail::timer_stop<graph_view_t::is_multi_gpu>(handle, hr_timer);
#endif

#ifdef TIMING
    detail::timer_start<graph_view_t::is_multi_gpu>(handle, hr_timer, "contract graph");
#endif

    // Count number of unique louvain clusters

    rmm::device_uvector<vertex_t> copied_louvain_partition(dendrogram->current_level_size(),
                                                           handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 dendrogram->current_level_begin(),
                 dendrogram->current_level_begin() + dendrogram->current_level_size(),
                 copied_louvain_partition.begin());
    auto nr_unique_louvain_clusters =
      remove_duplicates<vertex_t, multi_gpu>(handle, copied_louvain_partition);

    bool terminate = (nr_unique_louvain_clusters == current_graph_view.number_of_vertices());

    rmm::device_uvector<vertex_t> refined_leiden_partition(0, handle.get_stream());
    std::pair<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> leiden_to_louvain_map{
      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
      rmm::device_uvector<vertex_t>(0, handle.get_stream())};

    if (!terminate) {
      // Refine the current partition
      thrust::copy(handle.get_thrust_policy(),
                   dendrogram->current_level_begin(),
                   dendrogram->current_level_begin() + dendrogram->current_level_size(),
                   louvain_assignment_for_vertices.begin());

      if constexpr (graph_view_t::is_multi_gpu) {
        update_edge_src_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.begin(),
                                 src_louvain_assignment_cache.mutable_view());
        update_edge_dst_property(handle,
                                 current_graph_view,
                                 louvain_assignment_for_vertices.begin(),
                                 dst_louvain_assignment_cache.mutable_view());
      }

      std::tie(refined_leiden_partition, leiden_to_louvain_map) =
        detail::refine_clustering(handle,
                                  rng_state,
                                  current_graph_view,
                                  current_edge_weight_view,
                                  total_edge_weight,
                                  resolution,
                                  theta,
                                  vertex_weights,
                                  std::move(cluster_keys),
                                  std::move(cluster_weights),
                                  std::move(louvain_assignment_for_vertices),
                                  src_vertex_weights_cache,
                                  src_louvain_assignment_cache,
                                  dst_louvain_assignment_cache);
    }

    // Clear buffer and contract the graph
    final_Q = detail::compute_modularity(handle,
                                         current_graph_view,
                                         current_edge_weight_view,
                                         src_louvain_assignment_cache,
                                         dst_louvain_assignment_cache,
                                         louvain_assignment_for_vertices,
                                         cluster_weights,
                                         total_edge_weight,
                                         resolution);

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

    if (!terminate) {
      src_louvain_assignment_cache.clear(handle);
      dst_louvain_assignment_cache.clear(handle);

      auto nr_unique_leiden = static_cast<vertex_t>(leiden_to_louvain_map.first.size());
      if (graph_view_t::is_multi_gpu) {
        nr_unique_leiden = host_scalar_allreduce(
          handle.get_comms(), nr_unique_leiden, raft::comms::op_t::SUM, handle.get_stream());
      }

      terminate = terminate || (nr_unique_leiden == current_graph_view.number_of_vertices());

      if (nr_unique_leiden < current_graph_view.number_of_vertices()) {
        // Create aggregate graph based on refined (leiden) partition
        std::optional<rmm::device_uvector<vertex_t>> numbering_map{std::nullopt};
        std::tie(coarse_graph, coarsen_graph_edge_weight, numbering_map) =
          coarsen_graph(handle,
                        current_graph_view,
                        current_edge_weight_view,
                        refined_leiden_partition.data(),
                        true);

        current_graph_view = coarse_graph.view();

        current_edge_weight_view =
          std::make_optional<edge_property_view_t<edge_t, weight_t const*>>(
            (*coarsen_graph_edge_weight).view());

        // FIXME: reconsider what's put into dendrogram->current_level_begin()
        // at what point in the code. I'm just going to overwrite it here,
        // so perhaps it should be in different structures until now

        // New approach, mimic Louvain, we'll store the Leiden results in the dendrogram
        raft::copy(dendrogram->current_level_begin(),
                   refined_leiden_partition.data(),
                   refined_leiden_partition.size(),
                   handle.get_stream());

        louvain_of_refined_graph.resize(current_graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());
        rmm::device_uvector<vertex_t> numeric_sequence(
          current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

        detail::sequence_fill(handle.get_stream(),
                              numeric_sequence.data(),
                              numeric_sequence.size(),
                              current_graph_view.local_vertex_partition_range_first());

        relabel<vertex_t, multi_gpu>(
          handle,
          std::make_tuple(static_cast<vertex_t const*>((*numbering_map).begin()),
                          static_cast<vertex_t const*>(numeric_sequence.begin())),
          (*numbering_map).size(),
          dendrogram->current_level_begin(),
          dendrogram->current_level_size(),
          false);

        raft::copy(louvain_of_refined_graph.begin(),
                   numbering_map->data(),
                   numbering_map->size(),
                   handle.get_stream());

        relabel<vertex_t, multi_gpu>(
          handle,
          std::make_tuple(static_cast<vertex_t const*>(leiden_to_louvain_map.first.begin()),
                          static_cast<vertex_t const*>(leiden_to_louvain_map.second.begin())),
          leiden_to_louvain_map.first.size(),
          louvain_of_refined_graph.data(),
          louvain_of_refined_graph.size(),
          false);

        // Relabel clusters so that each cluster is identified by the lowest vertex id
        // that is assigned to it.  Note that numbering_map and numeric_sequence go out
        // of scope at the end of this block, we will reuse their memory
        raft::copy(numbering_map->begin(),
                   louvain_of_refined_graph.data(),
                   louvain_of_refined_graph.size(),
                   handle.get_stream());

        thrust::sort(handle.get_thrust_policy(),
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
          std::tie(*numbering_map, numeric_sequence) = shuffle_ext_vertex_value_pairs(
            handle, std::move(*numbering_map), std::move(numeric_sequence));

          thrust::sort(handle.get_thrust_policy(),
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

        relabel<vertex_t, multi_gpu>(
          handle,
          std::make_tuple(static_cast<vertex_t const*>((*numbering_map).begin()),
                          static_cast<vertex_t const*>(numeric_sequence.begin())),
          (*numbering_map).size(),
          louvain_of_refined_graph.data(),
          louvain_of_refined_graph.size(),
          false);
      }
    } else {
      // Reset dendrogram.
      //    FIXME: Revisit how dendrogram is populated
      detail::sequence_fill(handle.get_stream(),
                            dendrogram->current_level_begin(),
                            dendrogram->current_level_size(),
                            current_graph_view.local_vertex_partition_range_first());
    }

    copied_louvain_partition.resize(0, handle.get_stream());
    copied_louvain_partition.shrink_to_fit(handle.get_stream());

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

template <typename vertex_t, bool multi_gpu>
void relabel_cluster_ids(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& unique_cluster_ids,
                         vertex_t* clustering,
                         size_t num_nodes)
{
  vertex_t local_cluster_id_first{0};

  // Get unique cluster id and shuffle
  remove_duplicates<vertex_t, multi_gpu>(handle, unique_cluster_ids);

  if constexpr (multi_gpu) {
    auto cluster_ids_size_per_rank = cugraph::host_scalar_allgather(
      handle.get_comms(), unique_cluster_ids.size(), handle.get_stream());

    std::vector<vertex_t> cluster_ids_starts(cluster_ids_size_per_rank.size());
    std::exclusive_scan(cluster_ids_size_per_rank.begin(),
                        cluster_ids_size_per_rank.end(),
                        cluster_ids_starts.begin(),
                        size_t{0});
    local_cluster_id_first = cluster_ids_starts[handle.get_comms().get_rank()];
  }

  rmm::device_uvector<vertex_t> numbering_indices(unique_cluster_ids.size(), handle.get_stream());
  detail::sequence_fill(handle.get_stream(),
                        numbering_indices.data(),
                        numbering_indices.size(),
                        local_cluster_id_first);

  relabel<vertex_t, multi_gpu>(
    handle,
    std::make_tuple(static_cast<vertex_t const*>(unique_cluster_ids.begin()),
                    static_cast<vertex_t const*>(numbering_indices.begin())),
    unique_cluster_ids.size(),
    clustering,
    num_nodes,
    false);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void flatten_leiden_dendrogram(raft::handle_t const& handle,
                               graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                               Dendrogram<vertex_t> const& dendrogram,
                               vertex_t* clustering)
{
  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        vertex_ids_v.begin(),
                        vertex_ids_v.size(),
                        graph_view.local_vertex_partition_range_first());

  partition_at_level<vertex_t, multi_gpu>(
    handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  size_t max_level,
  weight_t resolution,
  weight_t theta = 1.0)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::leiden(
    handle, rng_state, graph_view, edge_weight_view, max_level, resolution, theta);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void flatten_leiden_dendrogram(raft::handle_t const& handle,
                               graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                               Dendrogram<vertex_t> const& dendrogram,
                               vertex_t* clustering)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  detail::flatten_leiden_dendrogram(handle, graph_view, dendrogram, clustering);
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
  weight_t theta = 1.0)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  detail::check_clustering(graph_view, clustering);

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram;
  weight_t modularity;

  std::tie(dendrogram, modularity) =
    detail::leiden(handle, rng_state, graph_view, edge_weight_view, max_level, resolution, theta);

  detail::flatten_leiden_dendrogram(handle, graph_view, *dendrogram, clustering);

  size_t local_num_verts = (*dendrogram).get_level_size_nocheck(0);
  rmm::device_uvector<vertex_t> unique_cluster_ids(local_num_verts, handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               clustering,
               clustering + local_num_verts,
               unique_cluster_ids.begin());

  detail::relabel_cluster_ids<vertex_t, multi_gpu>(
    handle, unique_cluster_ids, clustering, local_num_verts);

  return std::make_pair(dendrogram->num_levels(), modularity);
}

}  // namespace cugraph

/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "prims/fill_edge_property.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_e_by_src_dst_key.cuh"
#include "prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <thrust/fill.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
bool bellman_ford(raft::handle_t const& handle,
                  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                  vertex_t source,
                  vertex_t* predecessors,
                  weight_t* distances)
{
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  graph_view_t current_graph_view(graph_view);

  // edge mask
  cugraph::edge_property_t<graph_view_t, bool> edge_masks_even(handle, current_graph_view);
  cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_even);

  cugraph::edge_property_t<graph_view_t, bool> edge_masks_odd(handle, current_graph_view);
  cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_odd);

  cugraph::transform_e(
    handle,
    current_graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    cugraph::edge_dummy_property_t{}.view(),
    [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
      return !(src == dst);  // mask out self-loop
    },
    edge_masks_even.mutable_view());

  current_graph_view.attach_edge_mask(edge_masks_even.view());

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<weight_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  auto val_first = thrust::make_zip_iterator(thrust::make_tuple(distances, predecessors));
  thrust::transform(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_first()),
    thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_last()),
    val_first,
    [source] __device__(auto v) {
      auto distance = invalid_distance;
      if (v == source) { distance = weight_t{0.0}; }
      return thrust::make_tuple(distance, invalid_vertex);
    });

  auto src_distance_cache =
    graph_view_t::is_multi_gpu
      ? edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view)
      : edge_src_property_t<graph_view_t, weight_t>(handle);

  rmm::device_uvector<vertex_t> local_vertices(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        local_vertices.begin(),
                        local_vertices.size(),
                        current_graph_view.local_vertex_partition_range_first());

  constexpr size_t bucket_idx_curr = 0;
  constexpr size_t bucket_idx_next = 1;
  constexpr size_t num_buckets     = 2;

  vertex_frontier_t<vertex_t, void, graph_view_t::is_multi_gpu, true> vertex_frontier(handle,
                                                                                      num_buckets);

  if (current_graph_view.in_local_vertex_partition_range_nocheck(source)) {
    vertex_frontier.bucket(bucket_idx_curr).insert(source);
  }

  rmm::device_uvector<vertex_t> enqueue_counter(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

  thrust::fill(
    handle.get_thrust_policy(), enqueue_counter.begin(), enqueue_counter.end(), vertex_t{0});

  vertex_t nr_times_in_queue = 0;
  while (true) {
    if constexpr (graph_view_t::is_multi_gpu) {
      cugraph::update_edge_src_property(handle,
                                        current_graph_view,
                                        vertex_frontier.bucket(bucket_idx_curr).begin(),
                                        vertex_frontier.bucket(bucket_idx_curr).end(),
                                        distances,
                                        src_distance_cache);
    }

    auto [new_frontier_vertex_buffer, distance_predecessor_buffer] =
      cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
        handle,
        current_graph_view,
        vertex_frontier.bucket(bucket_idx_curr),
        graph_view_t::is_multi_gpu
          ? src_distance_cache.view()
          : detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances),
        edge_dst_dummy_property_t{}.view(),
        edge_weight_view,
        [distances,
         v_first = current_graph_view.local_vertex_partition_range_first(),
         v_last =
           current_graph_view.local_vertex_partition_range_last()] __device__(auto src,
                                                                              auto dst,
                                                                              auto src_dist,
                                                                              thrust::nullopt_t,
                                                                              auto wt) {
          assert(dst < v_first || dst >= v_last);

          auto dst_dist = distances[dst - v_first];
          auto relax    = (dst_dist > (src_dist + wt));

          return relax ? thrust::optional<thrust::tuple<weight_t, vertex_t>>{thrust::make_tuple(
                           src_dist + wt, src)}
                       : thrust::nullopt;
        },
        reduce_op::minimum<thrust::tuple<weight_t, vertex_t>>(),
        true);
    size_t nr_of_updated_vertices = new_frontier_vertex_buffer.size();

    if (graph_view_t::is_multi_gpu) {
      nr_of_updated_vertices = host_scalar_allreduce(
        handle.get_comms(), nr_of_updated_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (nr_of_updated_vertices == 0) { break; }

    thrust::for_each(handle.get_thrust_policy(),
                     new_frontier_vertex_buffer.begin(),
                     new_frontier_vertex_buffer.end(),
                     [v_first         = current_graph_view.local_vertex_partition_range_first(),
                      v_last          = current_graph_view.local_vertex_partition_range_last(),
                      enqueue_counter = enqueue_counter.begin()] __device__(vertex_t v) {
                       assert(v < v_first || v >= v_last);
                       enqueue_counter[v - v_first] += 1;
                     });

    nr_times_in_queue =
      thrust::count_if(handle.get_thrust_policy(),
                       enqueue_counter.begin(),
                       enqueue_counter.end(),
                       [nr_vertices = current_graph_view.number_of_vertices()] __device__(
                         auto freq_v) { return freq_v >= nr_vertices; });

    if (graph_view_t::is_multi_gpu) {
      nr_times_in_queue = host_scalar_allreduce(
        handle.get_comms(), nr_times_in_queue, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (nr_times_in_queue > 0) { break; }

    update_v_frontier(handle,
                      current_graph_view,
                      std::move(new_frontier_vertex_buffer),
                      std::move(distance_predecessor_buffer),
                      vertex_frontier,
                      std::vector<size_t>{bucket_idx_next},
                      distances,
                      thrust::make_zip_iterator(thrust::make_tuple(distances, predecessors)),
                      [] __device__(auto v, auto v_val, auto pushed_val) {
                        auto new_dist = thrust::get<0>(pushed_val);
                        auto update   = (new_dist < v_val);
                        return thrust::make_tuple(
                          update ? thrust::optional<size_t>{bucket_idx_next} : thrust::nullopt,
                          update ? thrust::optional<thrust::tuple<weight_t, vertex_t>>{pushed_val}
                                 : thrust::nullopt);
                      });

    vertex_frontier.bucket(bucket_idx_curr).clear();
    vertex_frontier.bucket(bucket_idx_curr).shrink_to_fit();

    if (vertex_frontier.bucket(bucket_idx_next).aggregate_size() > 0) {
      vertex_frontier.swap_buckets(bucket_idx_curr, bucket_idx_next);
    } else {
      break;
    }
  }

  if (nr_times_in_queue > 0) { return false; }
  return true;
}
}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
bool bellman_ford(raft::handle_t const& handle,
                  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                  vertex_t source,
                  vertex_t* predecessors,
                  weight_t* distances)
{
  return detail::bellman_ford(
    handle, graph_view, edge_weight_view, source, predecessors, distances);
}

}  // namespace cugraph

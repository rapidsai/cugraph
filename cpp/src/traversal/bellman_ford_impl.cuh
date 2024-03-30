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
void bellman_ford(raft::handle_t const& handle,
                  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
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

  bool debug_flag = current_graph_view.number_of_vertices() <= 7;

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

  // auto src_predecessor_cache =
  //   graph_view_t::is_multi_gpu
  //     ? edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view)
  //     : edge_src_property_t<graph_view_t, vertex_t>(handle);

  auto src_distance_cache =
    graph_view_t::is_multi_gpu
      ? edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view)
      : edge_src_property_t<graph_view_t, weight_t>(handle);

  // auto dst_predecessor_cache =
  //   graph_view_t::is_multi_gpu
  //     ? edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view)
  //     : edge_dst_property_t<graph_view_t, vertex_t>(handle);

  // auto dst_distance_cache =
  //   graph_view_t::is_multi_gpu
  //     ? edge_dst_property_t<graph_view_t, weight_t>(handle, current_graph_view)
  //     : edge_dst_property_t<graph_view_t, weight_t>(handle);

  // auto dst_key_cache = graph_view_t::is_multi_gpu
  //                        ? edge_dst_property_t<graph_view_t, vertex_t>(handle,
  //                        current_graph_view) : edge_dst_property_t<graph_view_t,
  //                        vertex_t>(handle);

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

  vertex_t itr_cnt           = 0;
  int nr_of_updated_vertices = 0;

  vertex_t nr_vertices_n_plus_times = 0;
  while (true) {
    if constexpr (graph_view_t::is_multi_gpu) {
      // update_edge_src_property(handle,
      //                          current_graph_view,
      //                          vertex_frontier.bucket(bucket_idx_curr).begin(),
      //                          vertex_frontier.bucket(bucket_idx_curr).end(),
      //                          predecessors,
      //                          src_predecessor_cache);
      update_edge_src_property(handle,
                               current_graph_view,
                               vertex_frontier.bucket(bucket_idx_curr).begin(),
                               vertex_frontier.bucket(bucket_idx_curr).end(),
                               distances,
                               src_distance_cache);

      // update_edge_dst_property(handle,
      //                          current_graph_view,
      //                          vertex_frontier.bucket(bucket_idx_curr).begin(),
      //                          vertex_frontier.bucket(bucket_idx_curr).end(),
      //                          predecessors,
      //                          dst_predecessor_cache);
      // update_edge_dst_property(handle,
      //                          current_graph_view,
      //                          vertex_frontier.bucket(bucket_idx_curr).begin(),
      //                          vertex_frontier.bucket(bucket_idx_curr).end(),
      //                          distances,
      //                          dst_distance_cache);

      // update_edge_dst_property(handle,
      //                          current_graph_view,
      //                          vertex_frontier.bucket(bucket_idx_curr).begin(),
      //                          vertex_frontier.bucket(bucket_idx_curr).end(),
      //                          local_vertices.begin(),
      //                          dst_key_cache);
    }

    /*
    auto src_input_property_values =
      graph_view_t::is_multi_gpu
        // ? view_concat(src_predecessor_cache.view(), src_distance_cache.view())
        ? src_distance_cache.view()
        // : view_concat(detail::edge_major_property_view_t<vertex_t, vertex_t
        // const*>(predecessors),
        //               detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances));
        : detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances);


    auto dst_input_property_values = graph_view_t::is_multi_gpu
      // ? view_concat(dst_predecessor_cache.view(), dst_distance_cache.view())
      ? dst_distance_cache.view()
        // : view_concat(
        //     detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(predecessors,
        //                                                                   vertex_t{0}),
        //     detail::edge_minor_property_view_t<vertex_t, weight_t const*>(distances,
        //     weight_t{0}));

        //     : view_concat(
        detail::edge_minor_property_view_t<vertex_t, weight_t const*>(distances, weight_t{0});
      */
    auto [new_frontier_vertex_buffer, distance_predecessor_buffer] =
      transform_reduce_v_frontier_outgoing_e_by_dst(
        handle,
        current_graph_view,
        vertex_frontier.bucket(bucket_idx_curr),
        graph_view_t::is_multi_gpu
          ? src_distance_cache.view()
          : detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances),
        edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        [debug_flag,
         distances,
         v_first = current_graph_view.local_vertex_partition_range_first(),
         v_last  = current_graph_view
                    .local_vertex_partition_range_last()] __device__(auto src,
                                                                     auto dst,
                                                                     auto src_dist,
                                                                     thrust::nullopt_t,
                                                                     // auto dst_dist,
                                                                     // thrust::tuple<vertex_t,
                                                                     // weight_t> src_pred_dist,
                                                                     // thrust::tuple<vertex_t,
                                                                     // weight_t> dst_pred_dist,
                                                                     auto wt) {
          if (dst < v_first || dst >= v_last) {
            printf("\n ****** dst = %d is not in this VP \n", static_cast<int>(dst));
          }
          // auto src_pred = thrust::get<0>(src_pred_dist);
          // auto src_dist = thrust::get<1>(src_pred_dist);

          // auto dst_pred = thrust::get<0>(dst_pred_dist);
          // auto dst_dist = thrust::get<1>(dst_pred_dist);

          auto dst_dist = distances[dst - v_first];

          /*src_pred = %d dst_pred = %d,*/
          if (debug_flag)
            printf("src = %d dst = %d wt = %f  src_dist = %f dst_dist = %f\n",
                   static_cast<int>(src),
                   static_cast<int>(dst),
                   static_cast<float>(wt),
                   // static_cast<int>(src_pred),
                   // static_cast<int>(dst_pred),
                   static_cast<float>(src_dist),
                   static_cast<float>(dst_dist));

          auto relax = (dst_dist > (src_dist + wt));

          return relax ? thrust::optional<thrust::tuple<weight_t, vertex_t>>{thrust::make_tuple(
                           src_dist + wt, src)}
                       : thrust::nullopt;
        },
        reduce_op::minimum<thrust::tuple<weight_t, vertex_t>>(),
        true);

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto new_frontier_vertex_buffer_title = std::string("nvb_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(new_frontier_vertex_buffer_title.c_str(),
                                  new_frontier_vertex_buffer.begin(),
                                  new_frontier_vertex_buffer.size(),
                                  std::cout);

      auto key_buffer = thrust::get<0>(
        cugraph::get_dataframe_buffer_cbegin(distance_predecessor_buffer).get_iterator_tuple());
      auto value_buffer = thrust::get<1>(
        cugraph::get_dataframe_buffer_cbegin(distance_predecessor_buffer).get_iterator_tuple());

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto key_buffer_title = std::string("key_buffer_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(
          key_buffer_title.c_str(), key_buffer, new_frontier_vertex_buffer.size(), std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto value_buffer_title = std::string("value_buffer_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(
          value_buffer_title.c_str(), value_buffer, new_frontier_vertex_buffer.size(), std::cout);
    }

    nr_of_updated_vertices = new_frontier_vertex_buffer.size();

    if (graph_view_t::is_multi_gpu) {
      nr_of_updated_vertices = host_scalar_allreduce(
        handle.get_comms(), nr_of_updated_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (nr_of_updated_vertices == 0) {
      std::cout << "no update break\n";
      break;
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto enqueue_counter_title = std::string("qc_b_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(enqueue_counter_title.c_str(),
                                  enqueue_counter.begin(),
                                  enqueue_counter.size(),
                                  std::cout);
    }

    thrust::for_each(handle.get_thrust_policy(),
                     new_frontier_vertex_buffer.begin(),
                     new_frontier_vertex_buffer.end(),
                     [v_first         = current_graph_view.local_vertex_partition_range_first(),
                      v_last          = current_graph_view.local_vertex_partition_range_last(),
                      enqueue_counter = enqueue_counter.begin()] __device__(vertex_t v) {
                       if (v < v_first || v >= v_last) {
                         printf("\n enque conter: *** v = %d is not in this VP \n",
                                static_cast<int>(v));
                       }
                       enqueue_counter[v - v_first] += 1;
                     });

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto enqueue_counter_title = std::string("qc_a_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(enqueue_counter_title.c_str(),
                                  enqueue_counter.begin(),
                                  enqueue_counter.size(),
                                  std::cout);
    }

    nr_vertices_n_plus_times = thrust::count_if(
      handle.get_thrust_policy(),
      enqueue_counter.begin(),
      enqueue_counter.end(),
      [n = current_graph_view.number_of_vertices()] __device__(auto flag) { return flag >= n; });

    if (graph_view_t::is_multi_gpu) {
      nr_vertices_n_plus_times = host_scalar_allreduce(
        handle.get_comms(), nr_vertices_n_plus_times, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (nr_vertices_n_plus_times > 0) {
      std::cout << "enque n+ break\n";
      break;
    }

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
      std::cout << "swap break\n";
      break;
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto local_vertices_title = std::string("local_vertices_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(
          local_vertices_title.c_str(), local_vertices.begin(), local_vertices.size(), std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto distances_title = std::string("distances_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(distances_title.c_str(),
                                  distances,
                                  current_graph_view.local_vertex_partition_range_size(),
                                  std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto predecessors_title = std::string("predecessors_").append(std::to_string(comm_rank));
      if (debug_flag)
        raft::print_device_vector(predecessors_title.c_str(),
                                  predecessors,
                                  current_graph_view.local_vertex_partition_range_size(),
                                  std::cout);
    }

    itr_cnt++;
    std::cout << "itr_cnt: " << itr_cnt << std::endl;
  }

  std::cout << "itr_cnt (out of loop) : " << itr_cnt << std::endl;

  if (nr_vertices_n_plus_times > 0) { std::cout << "Found -ve cycle " << std::endl; }
  // if ((itr_cnt == current_graph_view.number_of_vertices()) && (nr_of_updated_vertices > 0)) {
  //   std::cout << "Detected -ve cycle.\n";
  // }
}
}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void bellman_ford(raft::handle_t const& handle,
                  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                  vertex_t source,
                  vertex_t* predecessors,
                  weight_t* distances)
{
  detail::bellman_ford(handle, graph_view, edge_weight_view, source, predecessors, distances);
  std::cout << " returning from cugraph::bellman\n";
}

}  // namespace cugraph

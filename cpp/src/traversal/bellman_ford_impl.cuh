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
#include "prims/update_edge_src_dst_property.cuh"

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

  edge_src_property_t<graph_view_t, vertex_t> src_predecessor_cache(handle);
  edge_src_property_t<graph_view_t, weight_t> src_distance_cache(handle);

  edge_dst_property_t<graph_view_t, vertex_t> dst_predecessor_cache(handle);
  edge_dst_property_t<graph_view_t, weight_t> dst_distance_cache(handle);

  edge_dst_property_t<graph_view_t, vertex_t> dst_key_cache(handle);

  rmm::device_uvector<vertex_t> local_vertices(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

  detail::sequence_fill(handle.get_stream(),
                        local_vertices.begin(),
                        local_vertices.size(),
                        current_graph_view.local_vertex_partition_range_first());

  // auto vertex_begin =
  //   thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_first());
  // auto vertex_end =
  //   thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_last());

  // thrust::copy(handle.get_thrust_policy(), vertex_begin, vertex_end, local_vertices.begin());

  vertex_t itr_cnt           = 0;
  int nr_of_updated_vertices = 0;
  while (itr_cnt < current_graph_view.number_of_vertices()) {
    if constexpr (graph_view_t::is_multi_gpu) {
      src_predecessor_cache =
        edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      src_distance_cache = edge_src_property_t<graph_view_t, weight_t>(handle, current_graph_view);

      dst_predecessor_cache =
        edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);
      dst_distance_cache = edge_dst_property_t<graph_view_t, weight_t>(handle, current_graph_view);

      dst_key_cache = edge_dst_property_t<graph_view_t, vertex_t>(handle, current_graph_view);

      update_edge_src_property(handle, current_graph_view, predecessors, src_predecessor_cache);
      update_edge_src_property(handle, current_graph_view, distances, src_distance_cache);

      update_edge_dst_property(handle, current_graph_view, predecessors, dst_predecessor_cache);
      update_edge_dst_property(handle, current_graph_view, distances, dst_distance_cache);

      update_edge_dst_property(handle, current_graph_view, local_vertices.begin(), dst_key_cache);
    }

    auto src_input_property_values =
      graph_view_t::is_multi_gpu
        ? view_concat(src_predecessor_cache.view(), src_distance_cache.view())
        : view_concat(detail::edge_major_property_view_t<vertex_t, vertex_t const*>(predecessors),
                      detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances));

    auto dst_input_property_values =
      graph_view_t::is_multi_gpu
        ? view_concat(dst_predecessor_cache.view(), dst_distance_cache.view())
        : view_concat(
            detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(predecessors,
                                                                          vertex_t{0}),
            detail::edge_minor_property_view_t<vertex_t, weight_t const*>(distances, weight_t{0}));

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto local_vertices_title = std::string("local_vertices_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        local_vertices_title.c_str(), local_vertices.begin(), local_vertices.size(), std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto distances_title = std::string("distances_").append(std::to_string(comm_rank));
      raft::print_device_vector(distances_title.c_str(),
                                distances,
                                current_graph_view.local_vertex_partition_range_size(),
                                std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto predecessors_title = std::string("predecessors_").append(std::to_string(comm_rank));
      raft::print_device_vector(predecessors_title.c_str(),
                                predecessors,
                                current_graph_view.local_vertex_partition_range_size(),
                                std::cout);
    }

    rmm::device_uvector<vertex_t> edge_reduced_dst_keys(0, handle.get_stream());
    rmm::device_uvector<weight_t> minimum_weights(0, handle.get_stream());
    rmm::device_uvector<vertex_t> closest_preds(0, handle.get_stream());

    std::forward_as_tuple(edge_reduced_dst_keys, std::tie(minimum_weights, closest_preds)) =
      cugraph::transform_reduce_e_by_dst_key(
        handle,
        current_graph_view,
        src_input_property_values,
        dst_input_property_values,
        *edge_weight_view,
        graph_view_t::is_multi_gpu ? dst_key_cache.view()
                                   : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                                       local_vertices.begin(), vertex_t{0}),
        [] __device__(auto src,
                      auto dst,
                      thrust::tuple<vertex_t, weight_t> src_pred_dist,
                      thrust::tuple<vertex_t, weight_t> dst_pred_dist,
                      auto wt) {
          auto src_pred = thrust::get<0>(src_pred_dist);
          auto src_dist = thrust::get<1>(src_pred_dist);

          auto dst_pred = thrust::get<0>(dst_pred_dist);
          auto dst_dist = thrust::get<1>(dst_pred_dist);

          printf(
            "src = %d dst = %d wt = %f src_pred = %d dst_pred = %d, src_dist = %f dst_dist = %f\n",
            static_cast<int>(src),
            static_cast<int>(dst),
            static_cast<float>(wt),
            static_cast<int>(src_pred),
            static_cast<int>(dst_pred),
            static_cast<float>(src_dist),
            static_cast<float>(dst_dist));

          auto relax = (src_dist < invalid_distance) && (dst_dist > (src_dist + wt));

          return relax ? thrust::make_tuple(src_dist + wt, src)
                       : thrust::make_tuple(invalid_distance, invalid_vertex);
        },
        thrust::make_tuple(invalid_distance, invalid_vertex),
        reduce_op::minimum<thrust::tuple<weight_t, vertex_t>>{},
        true);

    if constexpr (graph_view_t::is_multi_gpu) {
      auto vertex_partition_range_lasts = current_graph_view.vertex_partition_range_lasts();

      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        vertex_partition_range_lasts.size(), handle.get_stream());

      raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream());

      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto func = cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());

      auto triplet_first = thrust::make_zip_iterator(
        edge_reduced_dst_keys.begin(), minimum_weights.begin(), closest_preds.begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        triplet_first,
        triplet_first + edge_reduced_dst_keys.size(),
        [func] __device__(auto val) { return func(thrust::get<0>(val)); },
        handle.get_comms().get_size(),
        std::numeric_limits<vertex_t>::max(),
        handle.get_stream());

      std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
      raft::update_host(h_tx_value_counts.data(),
                        d_tx_value_counts.data(),
                        d_tx_value_counts.size(),
                        handle.get_stream());
      handle.sync_stream();

      std::forward_as_tuple(std::tie(edge_reduced_dst_keys, minimum_weights, closest_preds),
                            std::ignore) =
        shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(
            edge_reduced_dst_keys.begin(), minimum_weights.begin(), closest_preds.begin()),
          h_tx_value_counts,
          handle.get_stream());
    }

    if (graph_view_t::is_multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_reduced_dst_keys_title =
        std::string("edge_reduced_dst_keys_").append(std::to_string(comm_rank));
      raft::print_device_vector(edge_reduced_dst_keys_title.c_str(),
                                edge_reduced_dst_keys.begin(),
                                edge_reduced_dst_keys.size(),
                                std::cout);

      auto minimum_weights_title =
        std::string("minimum_weights_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        minimum_weights_title.c_str(), minimum_weights.begin(), minimum_weights.size(), std::cout);

      auto closest_preds_title = std::string("closest_preds_").append(std::to_string(comm_rank));
      raft::print_device_vector(
        closest_preds_title.c_str(), closest_preds.begin(), closest_preds.size(), std::cout);
    }

    using flag_t                        = uint8_t;
    rmm::device_uvector<flag_t> updated = rmm::device_uvector<flag_t>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), updated.begin(), updated.end(), flag_t{false});

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(
        edge_reduced_dst_keys.begin(), minimum_weights.begin(), closest_preds.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
        edge_reduced_dst_keys.end(), minimum_weights.end(), closest_preds.end())),
      [distances,
       predecessors,
       updated = updated.begin(),
       v_first = current_graph_view.local_vertex_partition_range_first(),
       v_last =
         current_graph_view.local_vertex_partition_range_last()] __device__(auto v_dist_pred) {
        auto v = thrust::get<0>(v_dist_pred);
        if ((v < v_first) || (v >= v_last)) {
          printf("%d > > > > > out of range [%d %d)\n",
                 static_cast<int>(v),
                 static_cast<int>(v_first),
                 static_cast<int>(v_last));
        }

        auto dist     = thrust::get<1>(v_dist_pred);
        auto pred     = thrust::get<2>(v_dist_pred);
        auto v_offset = v - v_first;

        printf(" vertex %d : [pred=%d dist = %f)\n",
               static_cast<int>(v),
               static_cast<int>(pred),
               static_cast<float>(dist));

        if (pred != invalid_vertex) {
          updated[v_offset]      = flag_t{true};
          distances[v_offset]    = dist;
          predecessors[v_offset] = pred;
        }
      });

    nr_of_updated_vertices =
      thrust::count(handle.get_thrust_policy(), updated.begin(), updated.end(), flag_t{true});

    if constexpr (graph_view_t::is_multi_gpu) {
      nr_of_updated_vertices = host_scalar_allreduce(
        handle.get_comms(), nr_of_updated_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }

    itr_cnt++;
    std::cout << "itr_cnt: " << itr_cnt << std::endl;

    if (nr_of_updated_vertices == 0) {
      std::cout << "No more updates\n";
      break;
    }
  }

  std::cout << "itr_cnt (out of loop) : " << itr_cnt << std::endl;

  if ((itr_cnt == current_graph_view.number_of_vertices()) && (nr_of_updated_vertices > 0)) {
    std::cout << "Detected -ve cycle.\n";
  }

  ///

  /*
  vertex_t color_id = 0;
  while (true) {
    using flag_t                                 = uint8_t;
    rmm::device_uvector<flag_t> is_vertex_in_mis = rmm::device_uvector<flag_t>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), is_vertex_in_mis.begin(), is_vertex_in_mis.end(), 0);

    if (current_graph_view.compute_number_of_edges(handle) == 0) { break; }

    cugraph::edge_src_property_t<graph_view_t, flag_t> src_mis_flags(handle, current_graph_view);
    cugraph::edge_dst_property_t<graph_view_t, flag_t> dst_mis_flags(handle, current_graph_view);

    cugraph::update_edge_src_property(
      handle, current_graph_view, is_vertex_in_mis.begin(), src_mis_flags);

    cugraph::update_edge_dst_property(
      handle, current_graph_view, is_vertex_in_mis.begin(), dst_mis_flags);

    if (color_id % 2 == 0) {
      cugraph::transform_e(
        handle,
        current_graph_view,
        src_mis_flags.view(),
        dst_mis_flags.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [color_id] __device__(
          auto src, auto dst, auto is_src_in_mis, auto is_dst_in_mis, thrust::nullopt_t) {
          return !((is_src_in_mis == uint8_t{true}) || (is_dst_in_mis == uint8_t{true}));
        },
        edge_masks_odd.mutable_view());

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_even);
      current_graph_view.attach_edge_mask(edge_masks_odd.view());
    } else {
      cugraph::transform_e(
        handle,
        current_graph_view,
        src_mis_flags.view(),
        dst_mis_flags.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [color_id] __device__(
          auto src, auto dst, auto is_src_in_mis, auto is_dst_in_mis, thrust::nullopt_t) {
          return !((is_src_in_mis == uint8_t{true}) || (is_dst_in_mis == uint8_t{true}));
        },
        edge_masks_even.mutable_view());

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(handle, current_graph_view, bool{false}, edge_masks_odd);
      current_graph_view.attach_edge_mask(edge_masks_even.view());
    }

    color_id++;
  }
  */
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
}

}  // namespace cugraph

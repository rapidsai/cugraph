/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <thrust/fill.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, weight_t> approximate_weighted_matching(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: input graph for approximate_weighted_matching must "
                  "need to be symmetric");

  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  graph_view_t current_graph_view(graph_view);
  if (current_graph_view.has_edge_mask()) { current_graph_view.clear_edge_mask(); }

  cugraph::edge_property_t<graph_view_t, bool> edge_masks_even(handle, current_graph_view);
  cugraph::fill_edge_property(
    handle, current_graph_view, edge_masks_even.mutable_view(), bool{false});
  cugraph::edge_property_t<graph_view_t, bool> edge_masks_odd(handle, current_graph_view);
  cugraph::fill_edge_property(
    handle, current_graph_view, edge_masks_odd.mutable_view(), bool{false});

  if (graph_view.has_edge_mask()) {
    current_graph_view.attach_edge_mask(*(graph_view.edge_mask_view()));
  }
  // Mask out self-loop
  cugraph::transform_e(
    handle,
    current_graph_view,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    cugraph::edge_dummy_property_t{}.view(),
    [] __device__(
      auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) {
      return !(src == dst);
    },
    edge_masks_even.mutable_view());

  if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
  current_graph_view.attach_edge_mask(edge_masks_even.view());

  auto constexpr invalid_partner = invalid_vertex_id<vertex_t>::value;
  rmm::device_uvector<weight_t> offers_from_partners(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

  rmm::device_uvector<vertex_t> partners(current_graph_view.local_vertex_partition_range_size(),
                                         handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), partners.begin(), partners.end(), invalid_partner);
  thrust::fill(handle.get_thrust_policy(),
               offers_from_partners.begin(),
               offers_from_partners.end(),
               weight_t{0.0});

  rmm::device_uvector<vertex_t> local_vertices(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  detail::sequence_fill(handle.get_stream(),
                        local_vertices.begin(),
                        local_vertices.size(),
                        current_graph_view.local_vertex_partition_range_first());

  edge_src_property_t<graph_view_t, vertex_t> src_key_cache(handle);
  cugraph::edge_src_property_t<graph_view_t, bool> src_match_flags(handle);
  cugraph::edge_dst_property_t<graph_view_t, bool> dst_match_flags(handle);

  if constexpr (graph_view_t::is_multi_gpu) {
    src_key_cache = edge_src_property_t<graph_view_t, vertex_t>(handle, current_graph_view);

    update_edge_src_property(
      handle, current_graph_view, local_vertices.begin(), src_key_cache.mutable_view());

    src_match_flags = cugraph::edge_src_property_t<graph_view_t, bool>(handle, current_graph_view);
    dst_match_flags = cugraph::edge_dst_property_t<graph_view_t, bool>(handle, current_graph_view);
  }

  vertex_t loop_counter = 0;
  while (true) {
    //
    // For each candidate vertex, find the best possible target
    //

    rmm::device_uvector<vertex_t> candidates(0, handle.get_stream());
    rmm::device_uvector<weight_t> offers_from_candidates(0, handle.get_stream());
    rmm::device_uvector<vertex_t> targets(0, handle.get_stream());

    // FIXME: This can be implemented more efficiently if per_v_transform_reduce_incoming|outgoing_e
    // is updated to support reduction on thrust::tuple.
    std::forward_as_tuple(candidates, std::tie(offers_from_candidates, targets)) =
      cugraph::transform_reduce_e_by_src_key(
        handle,
        current_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_weight_view,
        graph_view_t::is_multi_gpu
          ? src_key_cache.view()
          : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(local_vertices.begin()),
        [] __device__(auto, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, auto wt) {
          return thrust::make_tuple(wt, dst);
        },
        thrust::make_tuple(weight_t{0.0}, invalid_partner),
        reduce_op::maximum<thrust::tuple<weight_t, vertex_t>>{},
        true);

    //
    // For each target, find the best offer
    //

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

      auto key_func = cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      std::forward_as_tuple(std::tie(candidates, offers_from_candidates, targets), std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(thrust::make_tuple(
            candidates.begin(), offers_from_candidates.begin(), targets.begin())),
          thrust::make_zip_iterator(
            thrust::make_tuple(candidates.end(), offers_from_candidates.end(), targets.end())),
          [key_func] __device__(auto val) { return key_func(thrust::get<2>(val)); },
          handle.get_stream());
    }

    auto itr_to_tuples = thrust::make_zip_iterator(
      thrust::make_tuple(offers_from_candidates.begin(), candidates.begin()));

    thrust::sort_by_key(handle.get_thrust_policy(), targets.begin(), targets.end(), itr_to_tuples);

    auto nr_unique_targets = thrust::count_if(handle.get_thrust_policy(),
                                              thrust::make_counting_iterator(size_t{0}),
                                              thrust::make_counting_iterator(targets.size()),
                                              is_first_in_run_t<vertex_t const*>{targets.data()});

    rmm::device_uvector<vertex_t> unique_targets(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<weight_t> best_offers_to_targets(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<vertex_t> best_candidates(nr_unique_targets, handle.get_stream());

    auto itr_to_reduced_tuples = thrust::make_zip_iterator(
      thrust::make_tuple(best_offers_to_targets.begin(), best_candidates.begin()));

    auto new_end = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      targets.begin(),
      targets.end(),
      itr_to_tuples,
      unique_targets.begin(),
      itr_to_reduced_tuples,
      thrust::equal_to<vertex_t>{},
      [] __device__(auto pair1, auto pair2) { return (pair1 > pair2) ? pair1 : pair2; });

    vertex_t nr_reduces_tuples =
      static_cast<vertex_t>(cuda::std::distance(unique_targets.begin(), new_end.first));

    targets                = std::move(unique_targets);
    offers_from_candidates = std::move(best_offers_to_targets);
    candidates             = std::move(best_candidates);

    //
    //  two vertex offer each other, that's a match
    //

    kv_store_t<vertex_t, vertex_t, false> target_candidate_map(targets.begin(),
                                                               targets.end(),
                                                               candidates.begin(),
                                                               invalid_vertex_id<vertex_t>::value,
                                                               invalid_vertex_id<vertex_t>::value,
                                                               handle.get_stream());

    rmm::device_uvector<vertex_t> candidates_of_candidates(0, handle.get_stream());

    if (graph_view_t::is_multi_gpu) {
      auto& comm       = handle.get_comms();
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
        vertex_partition_range_lasts.size(), handle.get_stream());

      raft::update_device(d_vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.data(),
                          vertex_partition_range_lasts.size(),
                          handle.get_stream());

      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> vertex_to_gpu_id_op{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size};

      candidates_of_candidates = cugraph::collect_values_for_keys(comm,
                                                                  target_candidate_map.view(),
                                                                  candidates.begin(),
                                                                  candidates.end(),
                                                                  vertex_to_gpu_id_op,
                                                                  handle.get_stream());
    } else {
      candidates_of_candidates.resize(candidates.size(), handle.get_stream());

      target_candidate_map.view().find(candidates.begin(),
                                       candidates.end(),
                                       candidates_of_candidates.begin(),
                                       handle.get_stream());
    }

    //
    // Mask out neighborhood of matched vertices
    //

    rmm::device_uvector<bool> is_vertex_matched = rmm::device_uvector<bool>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), is_vertex_matched.begin(), is_vertex_matched.end(), bool{false});

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(candidates_of_candidates.begin(),
                                                   targets.begin(),
                                                   candidates.begin(),
                                                   offers_from_candidates.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(candidates_of_candidates.end(),
                                                   targets.end(),
                                                   candidates.end(),
                                                   offers_from_candidates.end())),
      [partners             = partners.begin(),
       offers_from_partners = offers_from_partners.begin(),
       is_vertex_matched =
         raft::device_span<bool>(is_vertex_matched.data(), is_vertex_matched.size()),
       v_first =
         current_graph_view.local_vertex_partition_range_first()] __device__(auto msrc_tgt) {
        auto candidate_of_candidate = thrust::get<0>(msrc_tgt);
        auto tgt                    = thrust::get<1>(msrc_tgt);
        auto candiate               = thrust::get<2>(msrc_tgt);
        auto offer_value            = thrust::get<3>(msrc_tgt);

        if (candidate_of_candidate != invalid_partner && candidate_of_candidate == tgt) {
          auto tgt_offset                  = tgt - v_first;
          is_vertex_matched[tgt_offset]    = true;
          partners[tgt_offset]             = candiate;
          offers_from_partners[tgt_offset] = offer_value;
        }
      });

    if (current_graph_view.compute_number_of_edges(handle) == 0) { break; }

    if constexpr (graph_view_t::is_multi_gpu) {
      cugraph::update_edge_src_property(
        handle, current_graph_view, is_vertex_matched.begin(), src_match_flags.mutable_view());
      cugraph::update_edge_dst_property(
        handle, current_graph_view, is_vertex_matched.begin(), dst_match_flags.mutable_view());
    }

    if (loop_counter % 2 == 0) {
      if constexpr (graph_view_t::is_multi_gpu) {
        cugraph::transform_e(
          handle,
          current_graph_view,
          src_match_flags.view(),
          dst_match_flags.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [] __device__(
            auto src, auto dst, auto is_src_matched, auto is_dst_matched, cuda::std::nullopt_t) {
            return !((is_src_matched == true) || (is_dst_matched == true));
          },
          edge_masks_odd.mutable_view());
      } else {
        cugraph::transform_e(
          handle,
          current_graph_view,
          detail::edge_major_property_view_t<vertex_t, bool const*>(is_vertex_matched.begin()),
          detail::edge_minor_property_view_t<vertex_t, bool const*>(is_vertex_matched.begin(),
                                                                    vertex_t{0}),
          cugraph::edge_dummy_property_t{}.view(),
          [] __device__(
            auto src, auto dst, auto is_src_matched, auto is_dst_matched, cuda::std::nullopt_t) {
            return !((is_src_matched == true) || (is_dst_matched == true));
          },
          edge_masks_odd.mutable_view());
      }

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(
        handle, current_graph_view, edge_masks_even.mutable_view(), bool{false});
      current_graph_view.attach_edge_mask(edge_masks_odd.view());
    } else {
      if constexpr (graph_view_t::is_multi_gpu) {
        cugraph::transform_e(
          handle,
          current_graph_view,
          src_match_flags.view(),
          dst_match_flags.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [] __device__(
            auto src, auto dst, auto is_src_matched, auto is_dst_matched, cuda::std::nullopt_t) {
            return !((is_src_matched == true) || (is_dst_matched == true));
          },
          edge_masks_even.mutable_view());
      } else {
        cugraph::transform_e(
          handle,
          current_graph_view,
          detail::edge_major_property_view_t<vertex_t, bool const*>(is_vertex_matched.begin()),
          detail::edge_minor_property_view_t<vertex_t, bool const*>(is_vertex_matched.begin(),
                                                                    vertex_t{0}),
          cugraph::edge_dummy_property_t{}.view(),
          [] __device__(
            auto src, auto dst, auto is_src_matched, auto is_dst_matched, cuda::std::nullopt_t) {
            return !((is_src_matched == true) || (is_dst_matched == true));
          },
          edge_masks_even.mutable_view());
      }

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(
        handle, current_graph_view, edge_masks_odd.mutable_view(), bool{false});
      current_graph_view.attach_edge_mask(edge_masks_even.view());
    }

    loop_counter++;
  }

  weight_t sum_matched_edge_weights = thrust::reduce(
    handle.get_thrust_policy(), offers_from_partners.begin(), offers_from_partners.end());

  if constexpr (graph_view_t::is_multi_gpu) {
    sum_matched_edge_weights = host_scalar_allreduce(
      handle.get_comms(), sum_matched_edge_weights, raft::comms::op_t::SUM, handle.get_stream());
  }

  return std::make_tuple(std::move(partners), sum_matched_edge_weights / 2.0);
}
}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, weight_t> approximate_weighted_matching(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  return detail::approximate_weighted_matching(handle, graph_view, edge_weight_view);
}

}  // namespace cugraph

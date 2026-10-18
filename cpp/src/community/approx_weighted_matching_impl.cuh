/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/shuffle_wrappers.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/prims/fill_edge_property.cuh>
#include <cugraph/prims/fill_v.cuh>
#include <cugraph/prims/make_initialized_edge_property.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/transform_reduce_e_by_src_dst_key.cuh>
#include <cugraph/prims/transform_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/core/handle.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, vertex_t>, weight_t> approximate_weighted_matching(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: input graph for approximate_weighted_matching must "
                  "need to be symmetric");

  auto current_graph_view = graph_view;

  auto edge_masks_even = make_initialized_edge_property(handle, current_graph_view, false);
  auto edge_masks_odd  = make_initialized_edge_property(handle, current_graph_view, false);

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
  vertex_property_t<vertex_t, weight_t> offers_from_partners(handle, current_graph_view);
  vertex_property_t<vertex_t, vertex_t> partners(handle, current_graph_view);
  fill_v(handle, current_graph_view, partners.mutable_view(), invalid_partner);
  fill_v(handle, current_graph_view, offers_from_partners.mutable_view(), weight_t{0.0});

  cugraph::edge_src_property_t<vertex_t, vertex_t> src_key_cache(handle);
  cugraph::edge_src_property_t<vertex_t, bool> src_match_flags(handle);
  cugraph::edge_dst_property_t<vertex_t, bool> dst_match_flags(handle);

  using src_key_iterator_t =
    std::conditional_t<multi_gpu, vertex_t const*, thrust::counting_iterator<vertex_t>>;

  detail::edge_endpoint_property_view_t<vertex_t, src_key_iterator_t, vertex_t>
    src_key_cache_view{};

  if constexpr (multi_gpu) {
    vertex_property_t<vertex_t, vertex_t> local_vertices(handle, current_graph_view);
    transform_v(handle, current_graph_view, cuda::std::identity{}, local_vertices.mutable_view());

    src_key_cache = cugraph::edge_src_property_t<vertex_t, vertex_t>(handle, current_graph_view);

    update_edge_src_property(
      handle, current_graph_view, local_vertices.view(), src_key_cache.mutable_view());
    src_key_cache_view = src_key_cache.view();

    src_match_flags = cugraph::edge_src_property_t<vertex_t, bool>(handle, current_graph_view);
    dst_match_flags = cugraph::edge_dst_property_t<vertex_t, bool>(handle, current_graph_view);
  } else {
    src_key_cache_view =
      detail::make_edge_endpoint_property_view<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                               src_key_iterator_t,
                                               vertex_t,
                                               true>(
        current_graph_view,
        thrust::make_counting_iterator(current_graph_view.local_vertex_partition_range_first()));
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
    // is updated to support reduction on cuda::std::tuple.
    std::forward_as_tuple(candidates, std::tie(offers_from_candidates, targets)) =
      cugraph::transform_reduce_e_by_src_key(
        handle,
        current_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_weight_view,
        src_key_cache_view,
        [] __device__(auto, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, auto wt) {
          return cuda::std::make_tuple(wt, dst);
        },
        cuda::std::make_tuple(weight_t{0.0}, invalid_partner),
        reduce_op::maximum<cuda::std::tuple<weight_t, vertex_t>>{},
        true);

    //
    // For each target, find the best offer
    //

    if constexpr (multi_gpu) {
      std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(candidates));
      vertex_properties.push_back(std::move(offers_from_candidates));

      std::tie(targets, vertex_properties) =
        shuffle_int_vertices(handle,
                             std::move(targets),
                             std::move(vertex_properties),
                             graph_view.vertex_partition_range_lasts());

      candidates = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      offers_from_candidates =
        std::move(std::get<rmm::device_uvector<weight_t>>(vertex_properties[1]));
    }

    auto itr_to_tuples =
      thrust::make_zip_iterator(offers_from_candidates.begin(), candidates.begin());

    thrust::sort_by_key(handle.get_thrust_policy(), targets.begin(), targets.end(), itr_to_tuples);

    auto nr_unique_targets = thrust::count_if(handle.get_thrust_policy(),
                                              thrust::make_counting_iterator(size_t{0}),
                                              thrust::make_counting_iterator(targets.size()),
                                              is_first_in_run_t<vertex_t const*>{targets.data()});

    rmm::device_uvector<vertex_t> unique_targets(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<weight_t> best_offers_to_targets(nr_unique_targets, handle.get_stream());
    rmm::device_uvector<vertex_t> best_candidates(nr_unique_targets, handle.get_stream());

    auto itr_to_reduced_tuples =
      thrust::make_zip_iterator(best_offers_to_targets.begin(), best_candidates.begin());

    auto new_end = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      targets.begin(),
      targets.end(),
      itr_to_tuples,
      unique_targets.begin(),
      itr_to_reduced_tuples,
      cuda::std::equal_to<vertex_t>{},
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

    if constexpr (multi_gpu) {
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

      candidates_of_candidates = cugraph::collect_values_for_keys(handle,
                                                                  target_candidate_map.view(),
                                                                  candidates.begin(),
                                                                  candidates.end(),
                                                                  vertex_to_gpu_id_op);
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

    vertex_property_t<vertex_t, bool> is_vertex_matched(handle, current_graph_view);
    fill_v(handle, current_graph_view, is_vertex_matched.mutable_view(), false);

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(candidates_of_candidates.begin(),
                                targets.begin(),
                                candidates.begin(),
                                offers_from_candidates.begin()),
      thrust::make_zip_iterator(candidates_of_candidates.end(),
                                targets.end(),
                                candidates.end(),
                                offers_from_candidates.end()),
      [partners = detail::vertex_property_device_view_t{partners.mutable_view()},
       offers_from_partners =
         detail::vertex_property_device_view_t{offers_from_partners.mutable_view()},
       is_vertex_matched = detail::vertex_property_device_view_t{is_vertex_matched.mutable_view()},
       v_first =
         current_graph_view.local_vertex_partition_range_first()] __device__(auto msrc_tgt) {
        auto candidate_of_candidate = cuda::std::get<0>(msrc_tgt);
        auto tgt                    = cuda::std::get<1>(msrc_tgt);
        auto candiate               = cuda::std::get<2>(msrc_tgt);
        auto offer_value            = cuda::std::get<3>(msrc_tgt);

        if (candidate_of_candidate != invalid_partner && candidate_of_candidate == tgt) {
          auto tgt_offset = tgt - v_first;
          is_vertex_matched.set(tgt_offset, true);
          partners.set(tgt_offset, candiate);
          offers_from_partners.set(tgt_offset, offer_value);
        }
      });

    if (current_graph_view.compute_number_of_edges(handle) == 0) { break; }

    if constexpr (multi_gpu) {
      cugraph::update_edge_src_property(
        handle, current_graph_view, is_vertex_matched.view(), src_match_flags.mutable_view());
      cugraph::update_edge_dst_property(
        handle, current_graph_view, is_vertex_matched.view(), dst_match_flags.mutable_view());
    }

    if (loop_counter % 2 == 0) {
      if constexpr (multi_gpu) {
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
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [is_vertex_matched =
             detail::vertex_property_device_view_t{
               is_vertex_matched.view()}] __device__(auto src,
                                                     auto dst,
                                                     cuda::std::nullopt_t,
                                                     cuda::std::nullopt_t,
                                                     cuda::std::nullopt_t) {
            auto is_src_matched = is_vertex_matched.get(src);
            auto is_dst_matched = is_vertex_matched.get(dst);
            return !((is_src_matched == true) || (is_dst_matched == true));
          },
          edge_masks_odd.mutable_view());
      }

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(
        handle, current_graph_view, edge_masks_even.mutable_view(), bool{false});
      current_graph_view.attach_edge_mask(edge_masks_odd.view());
    } else {
      if constexpr (multi_gpu) {
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
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [is_vertex_matched =
             detail::vertex_property_device_view_t{
               is_vertex_matched.view()}] __device__(auto src,
                                                     auto dst,
                                                     cuda::std::nullopt_t,
                                                     cuda::std::nullopt_t,
                                                     cuda::std::nullopt_t) {
            auto is_src_matched = is_vertex_matched.get(src);
            auto is_dst_matched = is_vertex_matched.get(dst);
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

  auto sum_matched_edge_weights =
    reduce_v(handle, current_graph_view, offers_from_partners.view(), weight_t{0.0});

  return std::make_tuple(std::move(partners), sum_matched_edge_weights / 2.0);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, vertex_t>, weight_t> approx_weighted_matching(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  return detail::approximate_weighted_matching(handle, graph_view, edge_weight_view);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, weight_t> approximate_weighted_matching(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  auto [partners, sum_matched_edge_weights] =
    cugraph::approx_weighted_matching(handle, graph_view, edge_weight_view);

  return std::make_tuple(partners.release_buffer(), sum_matched_edge_weights);
}

}  // namespace cugraph

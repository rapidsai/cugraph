/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/prims/fill_edge_property.cuh>
#include <cugraph/prims/fill_v.cuh>
#include <cugraph/prims/make_initialized_edge_property.cuh>
#include <cugraph/prims/scatter_v.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/vertex_property.hpp>
#include <cugraph/vertex_property_device_view.cuh>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <thrust/iterator/constant_iterator.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
vertex_property_t<vertex_t, vertex_t> vertex_coloring(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  graph_view_t current_graph_view(graph_view);

  // edge mask
  auto edge_masks_even = make_initialized_edge_property(handle, current_graph_view, false);
  auto edge_masks_odd  = make_initialized_edge_property(handle, current_graph_view, false);

  cugraph::transform_e(
    handle,
    current_graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    cugraph::edge_dummy_property_t{}.view(),
    [] __device__(
      auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) {
      return !(src == dst);  // mask out self-loop
    },
    edge_masks_even.mutable_view());

  current_graph_view.attach_edge_mask(edge_masks_even.view());

  vertex_property_t<vertex_t, vertex_t> colors(handle, current_graph_view);
  fill_v(handle, current_graph_view, colors.mutable_view(), std::numeric_limits<vertex_t>::max());

  vertex_t color_id = 0;
  while (true) {
    auto mis = cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
      handle, current_graph_view, rng_state);

    vertex_property_t<vertex_t, bool> is_vertex_in_mis(handle, current_graph_view);
    fill_v(handle, current_graph_view, is_vertex_in_mis.mutable_view(), false);

    scatter_v(handle,
              current_graph_view,
              thrust::make_constant_iterator(true),
              thrust::make_constant_iterator(true) + mis.size(),
              mis.begin(),
              is_vertex_in_mis.mutable_view());

    auto new_color_first = cuda::make_transform_iterator(
      mis.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [color_id,
         colors  = detail::vertex_property_device_view_t{colors.view()},
         v_first = current_graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
          return cuda::std::min(color_id, colors.get(v - v_first));
        }));
    scatter_v(handle,
              current_graph_view,
              new_color_first,
              new_color_first + mis.size(),
              mis.begin(),
              colors.mutable_view());

    if (current_graph_view.compute_number_of_edges(handle) == 0) { break; }

    cugraph::edge_src_property_t<vertex_t, bool> src_mis_flags(handle);
    cugraph::edge_dst_property_t<vertex_t, bool> dst_mis_flags(handle);

    if constexpr (graph_view_t::is_multi_gpu) {
      src_mis_flags = cugraph::edge_src_property_t<vertex_t, bool>(handle, current_graph_view);
      dst_mis_flags = cugraph::edge_dst_property_t<vertex_t, bool>(handle, current_graph_view);

      cugraph::update_edge_src_property(
        handle, current_graph_view, is_vertex_in_mis.view(), src_mis_flags.mutable_view());

      cugraph::update_edge_dst_property(
        handle, current_graph_view, is_vertex_in_mis.view(), dst_mis_flags.mutable_view());
    }

    if (color_id % 2 == 0) {
      cugraph::transform_e(
        handle,
        current_graph_view,
        graph_view_t::is_multi_gpu
          ? src_mis_flags.view()
          : make_edge_src_property_view(current_graph_view, is_vertex_in_mis.view()),
        graph_view_t::is_multi_gpu
          ? dst_mis_flags.view()
          : make_edge_dst_property_view(current_graph_view, is_vertex_in_mis.view()),
        cugraph::edge_dummy_property_t{}.view(),
        [color_id] __device__(
          auto src, auto dst, auto is_src_in_mis, auto is_dst_in_mis, cuda::std::nullopt_t) {
          return !(is_src_in_mis || is_dst_in_mis);
        },
        edge_masks_odd.mutable_view());

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(
        handle, current_graph_view, edge_masks_even.mutable_view(), bool{false});
      current_graph_view.attach_edge_mask(edge_masks_odd.view());
    } else {
      cugraph::transform_e(
        handle,
        current_graph_view,
        graph_view_t::is_multi_gpu
          ? src_mis_flags.view()
          : make_edge_src_property_view(current_graph_view, is_vertex_in_mis.view()),
        graph_view_t::is_multi_gpu
          ? dst_mis_flags.view()
          : make_edge_dst_property_view(current_graph_view, is_vertex_in_mis.view()),
        cugraph::edge_dummy_property_t{}.view(),
        [color_id] __device__(
          auto src, auto dst, auto is_src_in_mis, auto is_dst_in_mis, cuda::std::nullopt_t) {
          return !(is_src_in_mis || is_dst_in_mis);
        },
        edge_masks_even.mutable_view());

      if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
      cugraph::fill_edge_property(
        handle, current_graph_view, edge_masks_odd.mutable_view(), bool{false});
      current_graph_view.attach_edge_mask(edge_masks_even.view());
    }

    color_id++;
  }
  return colors;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
vertex_property_t<vertex_t, vertex_t> vertex_coloring(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state,
  bool do_expensive_check)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  return detail::vertex_coloring(handle, graph_view, rng_state);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> vertex_coloring(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  return vertex_coloring(handle, graph_view, rng_state, false).release_buffer();
}

}  // namespace cugraph

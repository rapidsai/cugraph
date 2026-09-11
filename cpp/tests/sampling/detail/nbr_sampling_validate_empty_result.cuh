/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nbr_sampling_validate.hpp"
#include "sampling/detail/temporal_sampling_utils.cuh"

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/prims/count_if_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <optional>

namespace cugraph {
namespace test {

namespace {

template <typename vertex_t, typename GraphViewType>
rmm::device_uvector<bool> mark_local_seeds(raft::handle_t const& handle,
                                           GraphViewType const& graph_view,
                                           raft::device_span<vertex_t const> starting_vertices)
{
  auto const num_local_vertices =
    static_cast<size_t>(graph_view.local_vertex_partition_range_size());

  rmm::device_uvector<bool> is_seed(num_local_vertices, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), is_seed.begin(), is_seed.end(), false);

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(starting_vertices.size()),
    [starting_vertices,
     range_first = graph_view.local_vertex_partition_range_first(),
     range_last  = graph_view.local_vertex_partition_range_last(),
     is_seed     = raft::device_span<bool>{is_seed.data(), is_seed.size()}] __device__(size_t i) {
      auto v = starting_vertices[i];
      if ((v < range_first) || (v >= range_last)) { return; }
      is_seed[static_cast<size_t>(v - range_first)] = true;
    });

  return is_seed;
}

}  // namespace

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
bool validate_sampling_empty_result(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> starting_vertices,
  size_t num_sampled_edges,
  bool exclude_seed_destinations)
{
  // There is nothing to justify unless the result is empty.  Every rank sees the same aggregated
  // count, so returning here keeps the collectives below balanced.
  if (num_sampled_edges > 0) { return true; }

  auto is_seed = mark_local_seeds(handle, graph_view, starting_vertices);

  cugraph::edge_src_property_t<vertex_t, bool> edge_src_is_seed(handle, graph_view);
  cugraph::update_edge_src_property(
    handle, graph_view, is_seed.begin(), edge_src_is_seed.mutable_view());

  size_t num_eligible{0};
  if (exclude_seed_destinations) {
    cugraph::edge_dst_property_t<vertex_t, bool> edge_dst_is_seed(handle, graph_view);
    cugraph::update_edge_dst_property(
      handle, graph_view, is_seed.begin(), edge_dst_is_seed.mutable_view());

    num_eligible = static_cast<size_t>(
      cugraph::count_if_e(handle,
                          graph_view,
                          edge_src_is_seed.view(),
                          edge_dst_is_seed.view(),
                          cugraph::edge_dummy_property_t{}.view(),
                          cuda::proclaim_return_type<bool>(
                            [] __device__(auto, auto, bool src_is_seed, bool dst_is_seed, auto) {
                              // Disjoint sampling seeds the visited set with the starting vertices,
                              // so an edge into a starting vertex was never selectable at hop 0.
                              // Excluding every starting vertex rather than only those sharing a
                              // label undercounts when a destination seeds a different label, which
                              // keeps the check one-sided.
                              return src_is_seed && !dst_is_seed;
                            })));
  } else {
    num_eligible = static_cast<size_t>(cugraph::count_if_e(
      handle,
      graph_view,
      edge_src_is_seed.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      cugraph::edge_dummy_property_t{}.view(),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto, auto, bool src_is_seed, auto, auto) { return src_is_seed; })));
  }

  if constexpr (multi_gpu) {
    num_eligible = cugraph::host_scalar_allreduce(
      handle.get_comms(), num_eligible, raft::comms::op_t::SUM, handle.get_stream());
  }

  return num_eligible == 0;
}

template <typename vertex_t,
          typename edge_t,
          typename time_stamp_t,
          bool store_transposed,
          bool multi_gpu>
bool validate_sampling_empty_result(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  cugraph::edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
  size_t num_sampled_edges,
  cugraph::temporal_sampling_comparison_t temporal_sampling_comparison)
{
  // There is nothing to justify unless the result is empty.  Every rank sees the same aggregated
  // count, so returning here keeps the collectives below balanced.
  if (num_sampled_edges > 0) { return true; }

  // Decreasing walks start the walk at the end bound and treat the start bound as a floor;
  // increasing walks do the reverse.  This mirrors the initial frontier the implementation builds.
  auto const decreasing = cugraph::detail::is_temporal_decreasing(temporal_sampling_comparison);
  auto const seed_window_starts =
    decreasing ? starting_vertex_end_times : starting_vertex_start_times;
  auto const seed_window_ends =
    decreasing ? starting_vertex_start_times : starting_vertex_end_times;

  auto const unbounded_start =
    cugraph::detail::unbounded_temporal_window_start<time_stamp_t>(temporal_sampling_comparison);
  auto const unbounded_end =
    cugraph::detail::unbounded_temporal_window_end<time_stamp_t>(temporal_sampling_comparison);

  auto const num_local_vertices =
    static_cast<size_t>(graph_view.local_vertex_partition_range_size());

  rmm::device_uvector<bool> is_seed(num_local_vertices, handle.get_stream());
  rmm::device_uvector<time_stamp_t> window_starts(num_local_vertices, handle.get_stream());
  rmm::device_uvector<time_stamp_t> window_ends(num_local_vertices, handle.get_stream());

  // Vertices that aren't starting vertices get an empty window: swapping the two unbounded
  // sentinels makes passes_temporal_filter reject every edge time under either direction of
  // comparison, so the source side only has to carry the two bounds and not an is-seed flag.
  thrust::fill(handle.get_thrust_policy(), is_seed.begin(), is_seed.end(), false);
  thrust::fill(
    handle.get_thrust_policy(), window_starts.begin(), window_starts.end(), unbounded_end);
  thrust::fill(handle.get_thrust_policy(), window_ends.begin(), window_ends.end(), unbounded_start);

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(starting_vertices.size()),
    [starting_vertices,
     seed_window_starts = seed_window_starts ? seed_window_starts->data() : nullptr,
     seed_window_ends   = seed_window_ends ? seed_window_ends->data() : nullptr,
     unbounded_start,
     unbounded_end,
     range_first   = graph_view.local_vertex_partition_range_first(),
     range_last    = graph_view.local_vertex_partition_range_last(),
     is_seed       = raft::device_span<bool>{is_seed.data(), is_seed.size()},
     window_starts = raft::device_span<time_stamp_t>{window_starts.data(), window_starts.size()},
     window_ends   = raft::device_span<time_stamp_t>{window_ends.data(),
                                                     window_ends.size()}] __device__(size_t i) {
      auto v = starting_vertices[i];
      if ((v < range_first) || (v >= range_last)) { return; }
      auto offset = static_cast<size_t>(v - range_first);
      // A vertex seeding several labels races here and keeps one of its windows.  That window still
      // belongs to a real (seed, label) pair, so the resulting count stays a valid lower bound.
      is_seed[offset] = true;
      window_starts[offset] =
        (seed_window_starts != nullptr) ? seed_window_starts[i] : unbounded_start;
      window_ends[offset] = (seed_window_ends != nullptr) ? seed_window_ends[i] : unbounded_end;
    });

  cugraph::edge_src_property_t<vertex_t, cuda::std::tuple<time_stamp_t, time_stamp_t>>
    edge_src_window(handle, graph_view);
  cugraph::update_edge_src_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(window_starts.begin(), window_ends.begin()),
    edge_src_window.mutable_view());

  cugraph::edge_dst_property_t<vertex_t, bool> edge_dst_is_seed(handle, graph_view);
  cugraph::update_edge_dst_property(
    handle, graph_view, is_seed.begin(), edge_dst_is_seed.mutable_view());

  auto num_eligible = static_cast<size_t>(cugraph::count_if_e(
    handle,
    graph_view,
    edge_src_window.view(),
    edge_dst_is_seed.view(),
    edge_start_time_view,
    cuda::proclaim_return_type<bool>(
      [temporal_sampling_comparison] __device__(
        auto, auto, auto src_window, bool dst_is_seed, time_stamp_t edge_time) {
        // Temporal sampling is always disjoint and seeds the visited set with the starting
        // vertices, so an edge into a starting vertex was never selectable at hop 0.
        if (dst_is_seed) { return false; }
        return cugraph::detail::passes_temporal_filter<time_stamp_t>(temporal_sampling_comparison,
                                                                     cuda::std::get<0>(src_window),
                                                                     cuda::std::get<1>(src_window),
                                                                     edge_time);
      })));

  if constexpr (multi_gpu) {
    num_eligible = cugraph::host_scalar_allreduce(
      handle.get_comms(), num_eligible, raft::comms::op_t::SUM, handle.get_stream());
  }

  return num_eligible == 0;
}

}  // namespace test
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/prims/fill_edge_src_dst_property.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>

#include <cuda/std/tuple>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>

#include <limits>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename time_stamp_t, bool multi_gpu>
void update_temporal_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  raft::device_span<vertex_t const> vertices,
  raft::device_span<time_stamp_t const> vertex_times,
  std::optional<raft::device_span<time_stamp_t const>> vertex_window_ends,
  edge_property_view_t<edge_t, uint32_t*, bool> edge_time_mask_view,
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  time_stamp_t const STARTING_TIME{std::numeric_limits<time_stamp_t>::min()};
  time_stamp_t const WINDOW_END{std::numeric_limits<time_stamp_t>::max()};

  // The edge property machinery only supports arithmetic (or thrust::tuple of arithmetic) value
  // types, so the per-source frontier time and window end are tracked as two separate properties
  // and concatenated into a single (time, window_end) view for the transform.
  edge_src_property_t<vertex_t, time_stamp_t> edge_src_time(handle, graph_view);
  edge_src_property_t<vertex_t, time_stamp_t> edge_src_window_end(handle, graph_view);

  fill_edge_src_property(handle, graph_view, edge_src_time.mutable_view(), STARTING_TIME);
  fill_edge_src_property(handle, graph_view, edge_src_window_end.mutable_view(), WINDOW_END);

  update_edge_src_property(handle,
                           graph_view,
                           vertices.begin(),
                           vertices.end(),
                           vertex_times.begin(),
                           edge_src_time.mutable_view());

  if (vertex_window_ends) {
    update_edge_src_property(handle,
                             graph_view,
                             vertices.begin(),
                             vertices.end(),
                             vertex_window_ends->begin(),
                             edge_src_window_end.mutable_view());
  }

  cugraph::transform_e(
    handle,
    graph_view,
    view_concat(edge_src_time.view(), edge_src_window_end.view()),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_start_time_view,
    [temporal_sampling_comparison] __device__(
      auto src, auto dst, auto src_state, auto, auto edge_start_time) {
      auto src_time       = cuda::std::get<0>(src_state);
      auto src_window_end = cuda::std::get<1>(src_state);
      bool result         = false;
      switch (temporal_sampling_comparison) {
        case temporal_sampling_comparison_t::STRICTLY_INCREASING:
          result = (edge_start_time > src_time) && (edge_start_time <= src_window_end);
          break;
        case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
          result = (edge_start_time >= src_time) && (edge_start_time <= src_window_end);
          break;
        case temporal_sampling_comparison_t::STRICTLY_DECREASING:
          result = (edge_start_time < src_time) && (edge_start_time >= src_window_end);
          break;
        case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
          result = (edge_start_time <= src_time) && (edge_start_time >= src_window_end);
          break;
      }
      return result;
    },
    edge_time_mask_view,
    false);
}

}  // namespace detail
}  // namespace cugraph

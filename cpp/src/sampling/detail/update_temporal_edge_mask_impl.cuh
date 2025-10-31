/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>

#include <thrust/scatter.h>

#include <limits>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename edge_time_t, bool multi_gpu>
void update_temporal_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_start_time_view,
  raft::device_span<vertex_t const> vertices,
  raft::device_span<edge_time_t const> vertex_times,
  edge_property_view_t<edge_t, uint32_t*, bool> edge_time_mask_view,
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  edge_time_t const STARTING_TIME{std::numeric_limits<edge_time_t>::min()};

  edge_src_property_t<edge_t, edge_time_t> edge_src_times(handle, graph_view);

  // FIXME: As a future optimization, could consider moving this fill function to
  // outside the outer loop in the calling function and simply call this before sampling
  // with the current values and after sampling (with the same vertex set) with START_TIME
  // for each value so that we can reset everything back more efficiently.
  fill_edge_src_property(handle, graph_view, edge_src_times.mutable_view(), STARTING_TIME);

  update_edge_src_property(handle,
                           graph_view,
                           vertices.begin(),
                           vertices.end(),
                           vertex_times.begin(),
                           edge_src_times.mutable_view());

  cugraph::transform_e(
    handle,
    graph_view,
    edge_src_times.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_start_time_view,
    [temporal_sampling_comparison] __device__(
      auto src, auto dst, auto src_time, auto, auto edge_start_time) {
      bool result = false;
      switch (temporal_sampling_comparison) {
        case temporal_sampling_comparison_t::STRICTLY_INCREASING:
          result = (edge_start_time > src_time);
          break;
        case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
          result = (edge_start_time >= src_time);
          break;
        case temporal_sampling_comparison_t::STRICTLY_DECREASING:
          result = (edge_start_time < src_time);
          break;
        case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
          result = (edge_start_time <= src_time);
          break;
      }
      return result;
    },
    edge_time_mask_view,
    false);
}

}  // namespace detail
}  // namespace cugraph

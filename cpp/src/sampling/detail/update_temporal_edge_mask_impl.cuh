/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
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
  edge_property_view_t<edge_t, uint32_t*, bool> edge_time_mask_view)
{
  edge_time_t const STARTING_TIME{std::numeric_limits<edge_time_t>::min()};

  edge_src_property_t<edge_t, edge_time_t> edge_src_times(handle, graph_view);

#if 0
  // FIXME:  This call to update_edge_src_property seems like what I want, but it
  //         doesn't work.
  update_edge_src_property(handle,
                           graph_view,
                           vertices.begin(),
                           vertices.end(),
                           vertex_times.begin(),
                           edge_src_times.mutable_view(),
                           true);
#else
  // This alternative does work.

  rmm::device_uvector<edge_time_t> local_frontier_vertex_times(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());
  cugraph::detail::scalar_fill(
    handle, local_frontier_vertex_times.data(), local_frontier_vertex_times.size(), STARTING_TIME);

  thrust::scatter(
    handle.get_thrust_policy(),
    vertex_times.begin(),
    vertex_times.end(),
    thrust::make_transform_iterator(
      vertices.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [vertex_partition = vertex_partition_device_view_t<vertex_t, multi_gpu>(
           graph_view.local_vertex_partition_view())] __device__(auto v) {
          return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
        })),
    local_frontier_vertex_times.begin());

  update_edge_src_property(
    handle, graph_view, local_frontier_vertex_times.begin(), edge_src_times.mutable_view());
#endif

  cugraph::transform_e(
    handle,
    graph_view,
    edge_src_times.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_start_time_view,
    [] __device__(auto src, auto dst, auto src_time, auto, auto edge_start_time) {
      return edge_start_time > src_time;
    },
    edge_time_mask_view,
    false);
}

}  // namespace detail
}  // namespace cugraph

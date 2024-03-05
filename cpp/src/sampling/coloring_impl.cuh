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
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <cuda/functional>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <iostream>
#include <string>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> coloring(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  graph_view_t current_graph_view(graph_view);

  // edge mask
  cugraph::edge_property_t<graph_view_t, bool> edge_masks(handle, current_graph_view);
  cugraph::fill_edge_property(handle, current_graph_view, bool{true}, edge_masks);

  // device vector to store colors of vertices
  rmm::device_uvector<vertex_t> colors = rmm::device_uvector<vertex_t>(
    current_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), colors.begin(), colors.end(), std::numeric_limits<vertex_t>::max());

  vertex_t color_id = 0;
  while (true) {
    auto mis = cugraph::maximal_independent_set<vertex_t, edge_t, multi_gpu>(
      handle, current_graph_view, rng_state);

    using flag_t = uint8_t;

    rmm::device_uvector<flag_t> is_vertex_in_mis = rmm::device_uvector<flag_t>(
      current_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), is_vertex_in_mis.begin(), is_vertex_in_mis.end(), 0);

    thrust::for_each(
      handle.get_thrust_policy(),
      mis.begin(),
      mis.end(),
      [color_id,
       colors           = colors.data(),
       is_vertex_in_mis = is_vertex_in_mis.data(),
       v_first = current_graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
        auto v_offset              = v - v_first;
        is_vertex_in_mis[v_offset] = flag_t{1};
        vertex_t initial_color_id  = colors[v_offset];
        colors[v_offset]           = (color_id < initial_color_id) ? color_id : initial_color_id;
      });

    cugraph::edge_src_property_t<graph_view_t, flag_t> src_mis_flags(handle, current_graph_view);
    cugraph::edge_dst_property_t<graph_view_t, flag_t> dst_mis_flags(handle, current_graph_view);

    cugraph::update_edge_src_property(
      handle, current_graph_view, is_vertex_in_mis.begin(), src_mis_flags);

    cugraph::update_edge_dst_property(
      handle, current_graph_view, is_vertex_in_mis.begin(), dst_mis_flags);

    if (current_graph_view.compute_number_of_edges(handle) == 0) { break; }
    cugraph::transform_e(
      handle,
      current_graph_view,
      src_mis_flags.view(),
      dst_mis_flags.view(),
      edge_masks.view(),
      [] __device__(
        auto src, auto dst, auto is_src_to_remove, auto is_dst_to_remove, auto current_mask) {
        return (is_src_to_remove == 0) && (is_dst_to_remove == 0);
      },
      edge_masks.mutable_view());

    if (current_graph_view.has_edge_mask()) current_graph_view.clear_edge_mask();
    current_graph_view.attach_edge_mask(edge_masks.view());
    color_id++;
  }
  return colors;
}
}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> coloring(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  return detail::coloring(handle, graph_view, rng_state);
}

}  // namespace cugraph
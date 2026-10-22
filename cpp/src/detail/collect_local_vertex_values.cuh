/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/scatter.hpp>
#include <cugraph/vertex_property.hpp>

#include <cuda/functional>
#include <cuda/iterator>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename value_t,
          bool store_transposed,
          bool multi_gpu>
vertex_property_t<vertex_t, value_t> collect_local_vertex_values_from_ext_vertex_value_pairs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  rmm::device_uvector<vertex_t>&& d_vertices,
  rmm::device_uvector<value_t>&& d_values,
  vertex_property_view_t<vertex_t, vertex_t const*> renumber_map,
  value_t default_value,
  bool do_expensive_check)
{
  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    std::tie(d_vertices, d_values, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
      comm,
      d_vertices.begin(),
      d_vertices.end(),
      d_values.begin(),
      cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
        comm_size, major_comm_size, minor_comm_size},
      handle.get_stream());
  }

  // Now I can renumber locally
  d_vertices = renumber_local_ext_vertices(
    handle, graph_view, std::move(d_vertices), renumber_map, do_expensive_check);

  auto vertex_iterator = cuda::make_transform_iterator(
    d_vertices.begin(),
    cuda::proclaim_return_type<vertex_t>(
      [local_vertex_first = graph_view.local_vertex_partition_range_first()] __device__(
        vertex_t v) { return v - local_vertex_first; }));

  rmm::device_uvector<value_t> d_local_values(graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
  cugraph::fill(
    handle.get_thrust_policy(), d_local_values.begin(), d_local_values.end(), default_value);

  cugraph::scatter(handle.get_thrust_policy(),
                   d_values.begin(),
                   d_values.end(),
                   vertex_iterator,
                   d_local_values.begin());

  return vertex_property_t<vertex_t, value_t>(std::move(d_local_values),
                                              graph_view.local_vertex_partition_range_size());
}

}  // namespace detail
}  // namespace cugraph

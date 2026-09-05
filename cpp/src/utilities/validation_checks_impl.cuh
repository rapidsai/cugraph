/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "validation_checks.hpp"

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <thrust/count.h>

namespace cugraph {

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> vertices)
{
  auto vertex_partition =
    vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

  auto num_invalid_vertices =
    thrust::count_if(handle.get_thrust_policy(),
                     vertices.begin(),
                     vertices.end(),
                     [vertex_partition] __device__(auto v) {
                       return !(vertex_partition.is_valid_vertex(v) &&
                                vertex_partition.in_local_vertex_partition_range_nocheck(v));
                     });
  if constexpr (multi_gpu) {
    num_invalid_vertices = cugraph::host_scalar_allreduce(
      handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
  }

  return num_invalid_vertices;
}

}  // namespace cugraph

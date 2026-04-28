/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/edge_partition_device_view_impl.cuh"
#include "traversal/od_shortest_distances_impl.cuh"

namespace cugraph {

// SG instantiation

template rmm::device_uvector<float> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  raft::device_span<int32_t const> origins,
  raft::device_span<int32_t const> destinations,
  float cutoff,
  bool do_expensive_check);

template rmm::device_uvector<double> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  raft::device_span<int32_t const> origins,
  raft::device_span<int32_t const> destinations,
  double cutoff,
  bool do_expensive_check);

using view_t = edge_partition_device_view_t<int32_t, int32_t, false>;
using od_extract_iter_t =
  cuda::transform_iterator<extract_v_t<int32_t, uint32_t, uint64_t>, uint64_t const*>;
template void view_t::compute_number_of_edges_with_mask_async<od_extract_iter_t>(
  raft::device_span<uint32_t const>,
  od_extract_iter_t,
  od_extract_iter_t,
  raft::device_span<size_t>,
  rmm::cuda_stream_view) const;

}  // namespace cugraph

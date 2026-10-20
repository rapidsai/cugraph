/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/renumber_utils_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT void unrenumber_local_int_edges<int64_t, false, true>(
  raft::handle_t const& handle,
  std::vector<int64_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int64_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  int64_t const* renumber_map_labels,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

template CUGRAPH_EXPORT void unrenumber_local_int_edges<int64_t, true, true>(
  raft::handle_t const& handle,
  std::vector<int64_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int64_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  int64_t const* renumber_map_labels,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int64_t> unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  rmm::device_uvector<int64_t>&& vertices,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int64_t> unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  rmm::device_uvector<int64_t>&& vertices,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
unrenumber_local_int_edges(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::vector<size_t> const& edgelist_partition_offsets,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
unrenumber_local_int_edges(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::vector<size_t> const& edgelist_partition_offsets,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

}  // namespace cugraph

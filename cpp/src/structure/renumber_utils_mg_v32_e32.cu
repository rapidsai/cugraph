/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/renumber_utils_impl.cuh"

namespace cugraph {

// SG instantiation

template void renumber_ext_vertices<int32_t, true>(raft::handle_t const& handle,
                                                   int32_t* vertices,
                                                   size_t num_vertices,
                                                   int32_t const* renumber_map_labels,
                                                   int32_t local_int_vertex_first,
                                                   int32_t local_int_vertex_last,
                                                   bool do_expensive_check);

template void renumber_local_ext_vertices<int32_t, true>(raft::handle_t const& handle,
                                                         int32_t* vertices,
                                                         size_t num_vertices,
                                                         int32_t const* renumber_map_labels,
                                                         int32_t local_int_vertex_first,
                                                         int32_t local_int_vertex_last,
                                                         bool do_expensive_check);

template void unrenumber_int_vertices<int32_t, true>(
  raft::handle_t const& handle,
  int32_t* vertices,
  size_t num_vertices,
  int32_t const* renumber_map_labels,
  raft::host_span<int32_t const> vertex_partition_range_lasts,
  bool do_expensive_check);

template void unrenumber_local_int_edges<int32_t, false, true>(
  raft::handle_t const& handle,
  std::vector<int32_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int32_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  int32_t const* renumber_map_labels,
  raft::host_span<int32_t const> vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

template void unrenumber_local_int_edges<int32_t, true, true>(
  raft::handle_t const& handle,
  std::vector<int32_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int32_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  int32_t const* renumber_map_labels,
  raft::host_span<int32_t const> vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

}  // namespace cugraph

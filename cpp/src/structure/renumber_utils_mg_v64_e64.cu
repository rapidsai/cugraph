/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/renumber_utils_impl.cuh"

namespace cugraph {

// SG instantiation

template void renumber_ext_vertices<int64_t, true>(raft::handle_t const& handle,
                                                   int64_t* vertices,
                                                   size_t num_vertices,
                                                   int64_t const* renumber_map_labels,
                                                   int64_t local_int_vertex_first,
                                                   int64_t local_int_vertex_last,
                                                   bool do_expensive_check);

template void renumber_local_ext_vertices<int64_t, true>(raft::handle_t const& handle,
                                                         int64_t* vertices,
                                                         size_t num_vertices,
                                                         int64_t const* renumber_map_labels,
                                                         int64_t local_int_vertex_first,
                                                         int64_t local_int_vertex_last,
                                                         bool do_expensive_check);

template void unrenumber_int_vertices<int64_t, true>(
  raft::handle_t const& handle,
  int64_t* vertices,
  size_t num_vertices,
  int64_t const* renumber_map_labels,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  bool do_expensive_check);

template void unrenumber_local_int_edges<int64_t, false, true>(
  raft::handle_t const& handle,
  std::vector<int64_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int64_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  int64_t const* renumber_map_labels,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

template void unrenumber_local_int_edges<int64_t, true, true>(
  raft::handle_t const& handle,
  std::vector<int64_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int64_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  int64_t const* renumber_map_labels,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check);

}  // namespace cugraph

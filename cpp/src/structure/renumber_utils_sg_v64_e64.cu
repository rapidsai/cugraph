/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/renumber_utils_impl.cuh"

namespace cugraph {

// SG instantiation

template void renumber_ext_vertices<int64_t, false>(raft::handle_t const& handle,
                                                    int64_t* vertices,
                                                    size_t num_vertices,
                                                    int64_t const* renumber_map_labels,
                                                    int64_t local_int_vertex_first,
                                                    int64_t local_int_vertex_last,
                                                    bool do_expensive_check);

template void renumber_local_ext_vertices<int64_t, false>(raft::handle_t const& handle,
                                                          int64_t* vertices,
                                                          size_t num_vertices,
                                                          int64_t const* renumber_map_labels,
                                                          int64_t local_int_vertex_first,
                                                          int64_t local_int_vertex_last,
                                                          bool do_expensive_check);

template void unrenumber_local_int_vertices<int64_t>(raft::handle_t const& handle,
                                                     int64_t* vertices,
                                                     size_t num_vertices,
                                                     int64_t const* renumber_map_labels,
                                                     int64_t local_int_vertex_first,
                                                     int64_t local_int_vertex_last,
                                                     bool do_expensive_check);

template void unrenumber_int_vertices<int64_t, false>(
  raft::handle_t const& handle,
  int64_t* vertices,
  size_t num_vertices,
  int64_t const* renumber_map_labels,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  bool do_expensive_check);

template void unrenumber_local_int_edges<int64_t, false, false>(
  raft::handle_t const& handle,
  int64_t* edgelist_srcs /* [INOUT] */,
  int64_t* edgelist_dsts /* [INOUT] */,
  size_t num_edgelist_edges,
  int64_t const* renumber_map_labels,
  int64_t num_vertices,
  bool do_expensive_check);

template void unrenumber_local_int_edges<int64_t, true, false>(raft::handle_t const& handle,
                                                               int64_t* edgelist_srcs /* [INOUT] */,
                                                               int64_t* edgelist_dsts /* [INOUT] */,
                                                               size_t num_edgelist_edges,
                                                               int64_t const* renumber_map_labels,
                                                               int64_t num_vertices,
                                                               bool do_expensive_check);

}  // namespace cugraph

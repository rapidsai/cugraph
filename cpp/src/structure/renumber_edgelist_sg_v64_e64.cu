/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/renumber_edgelist_impl.cuh"

namespace cugraph {

// SG instantiation

template std::tuple<rmm::device_uvector<int64_t>, renumber_meta_t<int64_t, int64_t, false>>
renumber_edgelist<int64_t, int64_t, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertices,
  int64_t* edgelist_srcs /* [INOUT] */,
  int64_t* edgelist_dsts /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool store_transposed,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

}  // namespace cugraph

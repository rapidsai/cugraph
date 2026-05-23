/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/renumber_edgelist_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int32_t>,
                                   renumber_meta_t<int32_t, int32_t, true>>
renumber_edgelist<int32_t, int32_t, true>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& local_vertices,
  std::vector<int32_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<int32_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<int32_t> const& edgelist_edge_counts,
  std::optional<std::vector<std::vector<int32_t>>> const& edgelist_intra_partition_segment_offsets,
  bool store_transposed,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

}  // namespace cugraph

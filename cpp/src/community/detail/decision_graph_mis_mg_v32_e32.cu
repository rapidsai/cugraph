/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "decision_graph_mis.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<int32_t> vertices_in_mis_from_decision_edgelist<int32_t, true>(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  raft::host_span<int32_t const> vertex_partition_range_lasts,
  rmm::device_uvector<int32_t>&& d_srcs,
  rmm::device_uvector<int32_t>&& d_dsts);

}  // namespace detail
}  // namespace cugraph

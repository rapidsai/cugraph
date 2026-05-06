/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

/**
 * @brief Build a decision graph from an edgelist, compute a maximal independent set of moves,
 *        relabel MIS vertices to original ids, and (multi-GPU) shuffle them to owning ranks.
 *
 * @param vertex_partition_range_lasts Used only when multi_gpu is true (shuffle_int_vertices).
 */
template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> vertices_in_mis_from_decision_edgelist(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  rmm::device_uvector<vertex_t>&& d_srcs,
  rmm::device_uvector<vertex_t>&& d_dsts);

}  // namespace detail
}  // namespace cugraph

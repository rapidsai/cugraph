/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "maximal_independent_moves.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<int64_t> maximal_independent_moves(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& decision_graph_view,
  raft::random::RngState& rng_state);

}  // namespace detail
}  // namespace cugraph

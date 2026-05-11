/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "maximal_independent_moves.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> maximal_independent_moves(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& decision_graph_view,
  raft::random::RngState& rng_state);

}  // namespace detail
}  // namespace cugraph

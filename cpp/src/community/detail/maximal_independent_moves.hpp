/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> maximal_independent_moves(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state);

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vertex_coloring_impl.cuh"

namespace cugraph {

template rmm::device_uvector<int64_t> vertex_coloring(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  raft::random::RngState& rng_state);

}  // namespace cugraph

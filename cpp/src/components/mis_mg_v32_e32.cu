/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mis_impl.cuh"
namespace cugraph {

template rmm::device_uvector<int32_t> maximal_independent_set(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  raft::random::RngState& rng_state);

}  // namespace cugraph

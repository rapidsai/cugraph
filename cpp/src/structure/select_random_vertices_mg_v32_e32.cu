/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/select_random_vertices_impl.hpp"

namespace cugraph {

template rmm::device_uvector<int32_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<raft::device_span<int32_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool do_expensive_check);

template rmm::device_uvector<int32_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<raft::device_span<int32_t const>> given_set,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices,
  bool do_expensive_check);

}  // namespace cugraph

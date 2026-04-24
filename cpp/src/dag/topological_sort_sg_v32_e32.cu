/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dag/topological_sort_impl.cuh"

namespace cugraph {

template rmm::device_uvector<int32_t> topological_sort(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  bool do_expensive_check);

}  // namespace cugraph

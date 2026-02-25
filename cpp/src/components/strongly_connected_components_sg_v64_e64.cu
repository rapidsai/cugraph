/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "components/strongly_connected_components_impl.cuh"

namespace cugraph {

// SG instantiations

template rmm::device_uvector<int64_t> strongly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  bool do_expensive_check);

}  // namespace cugraph

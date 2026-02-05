/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "components/weakly_connected_components_impl.cuh"

namespace cugraph {

// SG instantiations

template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  int64_t* components,
  bool do_expensive_check);

}  // namespace cugraph

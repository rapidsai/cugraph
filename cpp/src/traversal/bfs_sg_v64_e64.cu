/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/bfs_impl.cuh"

namespace cugraph {

// SG instantiation

template void bfs(raft::handle_t const& handle,
                  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                  int64_t* distances,
                  int64_t* predecessors,
                  int64_t const* sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int64_t depth_limit,
                  bool do_expensive_check);

}  // namespace cugraph

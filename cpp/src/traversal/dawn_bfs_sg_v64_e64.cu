/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/dawn_bfs_impl.cuh"

namespace cugraph {

template void dawn_bfs(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                       int64_t* distances,
                       int64_t const* sources,
                       size_t n_sources,
                       int64_t depth_limit,
                       bool do_expensive_check);

}  // namespace cugraph

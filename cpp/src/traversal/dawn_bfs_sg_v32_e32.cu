/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/dawn_bfs_impl.cuh"

namespace cugraph {

template void dawn_bfs(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                       int32_t* distances,
                       int32_t const* sources,
                       size_t n_sources,
                       int32_t depth_limit,
                       bool do_expensive_check);

}  // namespace cugraph

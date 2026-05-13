/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/bfs_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT void bfs(raft::handle_t const& handle,
                                 graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                                 int32_t* distances,
                                 int32_t* predecessors,
                                 int32_t const* sources,
                                 size_t n_sources,
                                 bool direction_optimizing,
                                 int32_t depth_limit,
                                 bool do_expensive_check);

}  // namespace cugraph

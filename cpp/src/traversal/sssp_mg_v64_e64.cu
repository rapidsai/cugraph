/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "traversal/sssp_impl.cuh"

namespace cugraph {

// MG instantiation

template void sssp(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                   edge_property_view_t<int64_t, float const*> edge_weight_view,
                   float* distances,
                   int64_t* predecessors,
                   int64_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                   edge_property_view_t<int64_t, double const*> edge_weight_view,
                   double* distances,
                   int64_t* predecessors,
                   int64_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

}  // namespace cugraph

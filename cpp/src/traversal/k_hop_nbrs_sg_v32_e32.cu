/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "traversal/k_hop_nbrs_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<int32_t>>
k_hop_nbrs(raft::handle_t const& handle,
           graph_view_t<int32_t, int32_t, false, false> const& graph_view,
           raft::device_span<int32_t const> start_vertices,
           size_t k,
           bool do_expensive_check);

}  // namespace cugraph

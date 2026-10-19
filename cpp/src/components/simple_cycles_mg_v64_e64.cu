/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "components/simple_cycles_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiations

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
simple_cycles(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, false, true> const& graph_view,
              std::optional<raft::device_span<int64_t const>> seed_vertices,
              int64_t length_bound,
              bool do_expensive_check);

}  // namespace cugraph

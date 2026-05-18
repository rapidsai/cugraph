/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "validation_checks_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

template CUGRAPH_EXPORT size_t
count_invalid_vertices(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                       raft::device_span<int64_t const> vertices);

template CUGRAPH_EXPORT size_t
count_invalid_vertices(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                       raft::device_span<int64_t const> vertices);

}  // namespace cugraph

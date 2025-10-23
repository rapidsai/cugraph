/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/triangle_count_impl.cuh"

namespace cugraph {

template void triangle_count(raft::handle_t const& handle,
                             graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                             std::optional<raft::device_span<int64_t const>> vertices,
                             raft::device_span<int64_t> counts,
                             bool do_expensive_check);

}  // namespace cugraph

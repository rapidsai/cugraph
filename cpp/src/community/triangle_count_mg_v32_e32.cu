/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/triangle_count_impl.cuh"

namespace cugraph {

template void triangle_count(raft::handle_t const& handle,
                             graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                             std::optional<raft::device_span<int32_t const>> vertices,
                             raft::device_span<int32_t> counts,
                             bool do_expensive_check);

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/update_visited_utils_impl.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<int32_t>>>
update_dst_visited_vertices_and_labels<int64_t, int64_t, false>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, false> const&,
  rmm::device_uvector<int64_t>&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  raft::device_span<int64_t const>,
  std::optional<raft::device_span<int32_t const>>);

}  // namespace detail
}  // namespace cugraph

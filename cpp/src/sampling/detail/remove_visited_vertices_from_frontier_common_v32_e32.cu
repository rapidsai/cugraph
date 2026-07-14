/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sampling/detail/remove_visited_vertices_from_frontier.cuh"

#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>

#include <optional>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int32_t>,
                                   std::optional<rmm::device_uvector<int32_t>>,
                                   std::optional<rmm::device_uvector<int32_t>>,
                                   std::optional<rmm::device_uvector<int32_t>>>
remove_visited_vertices_from_frontier(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& frontier_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& frontier_vertex_labels,
  std::optional<rmm::device_uvector<int32_t>>&& frontier_vertex_times,
  std::optional<rmm::device_uvector<int32_t>>&& frontier_vertex_window_ends,
  raft::device_span<int32_t const> vertices_used_as_source,
  std::optional<raft::device_span<int32_t const>> vertex_labels_used_as_source);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int32_t>,
                                   std::optional<rmm::device_uvector<int32_t>>,
                                   std::optional<rmm::device_uvector<int64_t>>,
                                   std::optional<rmm::device_uvector<int64_t>>>
remove_visited_vertices_from_frontier(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& frontier_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& frontier_vertex_labels,
  std::optional<rmm::device_uvector<int64_t>>&& frontier_vertex_times,
  std::optional<rmm::device_uvector<int64_t>>&& frontier_vertex_window_ends,
  raft::device_span<int32_t const> vertices_used_as_source,
  std::optional<raft::device_span<int32_t const>> vertex_labels_used_as_source);

}  // namespace detail
}  // namespace cugraph

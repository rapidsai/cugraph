/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/triangle_count_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

template CUGRAPH_EXPORT
  std::variant<vertex_property_t<int32_t, int32_t>, rmm::device_uvector<int32_t>>
  triangle_count(raft::handle_t const& handle,
                 graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                 std::optional<raft::device_span<int32_t const>> vertices,
                 bool do_expensive_check);

template CUGRAPH_EXPORT void triangle_count(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<raft::device_span<int32_t const>> vertices,
  raft::device_span<int32_t> counts,
  bool do_expensive_check);

}  // namespace cugraph

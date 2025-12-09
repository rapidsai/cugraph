/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "validation_checks_impl.cuh"

namespace cugraph {

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  raft::device_span<int32_t const> vertices);

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  raft::device_span<int32_t const> vertices);

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  raft::device_span<int32_t const> vertices);

template size_t count_invalid_vertices(raft::handle_t const& handle,
                                       graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                                       raft::device_span<int32_t const> vertices);

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  raft::device_span<int64_t const> vertices);

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  raft::device_span<int64_t const> vertices);

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  raft::device_span<int64_t const> vertices);

template size_t count_invalid_vertices(raft::handle_t const& handle,
                                       graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                                       raft::device_span<int64_t const> vertices);

}  // namespace cugraph

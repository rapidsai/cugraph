/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/graph_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

namespace cugraph {

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
size_t count_invalid_vertices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> vertices);

}

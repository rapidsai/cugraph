/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void random_walks_validate(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>,
  rmm::device_uvector<vertex_t>&& d_start,
  rmm::device_uvector<vertex_t>&& d_vertices,
  std::optional<rmm::device_uvector<weight_t>>&& d_weights,
  size_t max_length);

}  // namespace test
}  // namespace cugraph

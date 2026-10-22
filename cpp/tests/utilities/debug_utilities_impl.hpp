/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utilities/debug_utilities.hpp"

#include <cugraph/graph_functions.hpp>

namespace cugraph {
namespace test {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
void print_edges(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<cugraph::vertex_property_view_t<vertex_t, vertex_t const*>> renumber_map)
{
  auto [srcs, dsts, edge_property_values] = cugraph::decompress_to_edgelist(
    handle,
    graph_view,
    edge_weight_view
      ? std::vector<cugraph::edge_arithmetic_property_view_t<edge_t>>{*edge_weight_view}
      : std::vector<cugraph::edge_arithmetic_property_view_t<edge_t>>{},
    renumber_map);
  raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
  raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
  if (edge_weight_view) {
    auto const& weights = std::get<rmm::device_uvector<weight_t>>(edge_property_values[0]);
    raft::print_device_vector("weights", weights.data(), weights.size(), std::cout);
  }
}

}  // namespace test
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
  std::optional<raft::device_span<vertex_t const>> renumber_map)
{
  auto [srcs, dsts, weights, edge_ids, edge_types] = cugraph::
    decompress_to_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle, graph_view, edge_weight_view, std::nullopt, std::nullopt, renumber_map);
  raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
  raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
  if (weights) {
    raft::print_device_vector("weights", (*weights).data(), (*weights).size(), std::cout);
  }
}

}  // namespace test
}  // namespace cugraph

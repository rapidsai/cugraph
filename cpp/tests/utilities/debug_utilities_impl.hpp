/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  auto [srcs, dsts, weights, edge_ids] =
    cugraph::decompress_to_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle, graph_view, edge_weight_view, std::nullopt, renumber_map);
  raft::print_device_vector("srcs", srcs.data(), srcs.size(), std::cout);
  raft::print_device_vector("dsts", dsts.data(), dsts.size(), std::cout);
  if (weights) {
    raft::print_device_vector("weights", (*weights).data(), (*weights).size(), std::cout);
  }
}

}  // namespace test
}  // namespace cugraph

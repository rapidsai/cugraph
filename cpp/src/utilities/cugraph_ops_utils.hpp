/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#pragma once

#include <cugraph/graph_view.hpp>

#include <cugraph-ops/graph/format.hpp>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename NodeTypeT, typename EdgeTypeT>
ops::gnn::graph::fg_csr<EdgeTypeT> get_graph(
  graph_view_t<NodeTypeT, EdgeTypeT, false, false> const& gview)
{
  ops::gnn::graph::fg_csr<EdgeTypeT> graph;
  graph.n_nodes   = gview.number_of_vertices();
  graph.n_indices = gview.number_of_edges();
  // FIXME: this is evil and is just temporary until we have a matching type in cugraph-ops
  // or we change the type accepted by the functions calling into cugraph-ops
  graph.offsets = const_cast<EdgeTypeT*>(gview.local_edge_partition_view().offsets().data());
  graph.indices = const_cast<EdgeTypeT*>(gview.local_edge_partition_view().indices().data());
  return graph;
}

template <typename NodeTypeT, typename EdgeTypeT>
std::tuple<ops::gnn::graph::fg_csr<EdgeTypeT>, NodeTypeT> get_graph_and_max_degree(
  graph_view_t<NodeTypeT, EdgeTypeT, false, false> const& gview)
{
  // FIXME this is sufficient for now, but if there is a fast (cached) way
  // of getting max degree, use that instead
  auto max_degree = std::numeric_limits<NodeTypeT>::max();
  return std::make_tuple(get_graph(gview), max_degree);
}

}  // namespace detail
}  // namespace cugraph

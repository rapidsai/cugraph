/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

namespace cugraph {
namespace detail {

template <typename NodeTypeT, typename EdgeTypeT>
ops::graph::csc<EdgeTypeT, NodeTypeT> get_graph(
  graph_view_t<NodeTypeT, EdgeTypeT, false, false> const& gview)
{
  ops::graph::csc<EdgeTypeT, NodeTypeT> graph;
  graph.n_src_nodes = gview.number_of_vertices();
  graph.n_dst_nodes = gview.number_of_vertices();
  graph.n_indices   = gview.number_of_edges();
  // FIXME this is sufficient for now, but if there is a fast (cached) way
  // of getting max degree, use that instead
  graph.dst_max_in_degree = std::numeric_limits<EdgeTypeT>::max();
  // FIXME: this is evil and is just temporary until we have a matching type in cugraph-ops
  // or we change the type accepted by the functions calling into cugraph-ops
  graph.offsets = const_cast<EdgeTypeT*>(gview.local_edge_partition_view().offsets().data());
  graph.indices = const_cast<EdgeTypeT*>(gview.local_edge_partition_view().indices().data());
  return graph;
}

}  // namespace detail
}  // namespace cugraph

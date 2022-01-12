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

#include <cugraph-ops/graph/format.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename IdxT, typename WeightT>
ops::gnn::graph::fg_csr<IdxT> get_graph(
  graph_view_t<IdxT, IdxT, WeightT, false, false> const& gview)
{
  ops::gnn::graph::fg_csr<IdxT> graph;
  graph.n_nodes = gview.get_number_of_vertices();
  graph.n_indices = gview.get_number_of_edges();
  graph.offsets = gview.get_matrix_partition_view().get_offsets();
  graph.indices = gview.get_matrix_partition_view().get_indices();
  return graph;
}

template <typename IdxT, typename WeightT>
std::tuple<ops::gnn::graph::fg_csr<IdxT>, IdxT> get_graph_and_max_degree(
  graph_view_t<IdxT, IdxT, WeightT, false, false> const& gview)
{
  // FIXME this is sufficient for now, but if there is a fast (cached) way
  // of getting max degree, use that instead
  int32_t max_degree = std::numeric_limits<int32_t>::max();
  return std::make_tuple(get_graph(gview), max_degree);
}

}  // namespace detail
}  // namespace cugraph

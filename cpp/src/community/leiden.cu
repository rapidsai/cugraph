/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <community/leiden.cuh>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<size_t, weight_t> leiden(raft::handle_t const &handle,
                                   GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                                   vertex_t *clustering,
                                   size_t max_level,
                                   weight_t resolution)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr,
                  "Invalid input argument: leiden expects a weighted graph");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");

  Leiden<GraphCSRView<vertex_t, edge_t, weight_t>> runner(handle, graph);
  weight_t wt = runner(max_level, resolution);

  runner.get_dendogram().partition_at_level(clustering, runner.get_dendogram().num_levels());

  // FIXME: Consider returning the Dendogram at some point
  return std::make_pair(runner.get_dendogram().num_levels(), wt);
}

// Explicit template instantations
template std::pair<size_t, float> leiden(
  raft::handle_t const &, GraphCSRView<int32_t, int32_t, float> const &, int32_t *, size_t, float);

template std::pair<size_t, double> leiden(raft::handle_t const &,
                                          GraphCSRView<int32_t, int32_t, double> const &,
                                          int32_t *,
                                          size_t,
                                          double);

}  // namespace cugraph

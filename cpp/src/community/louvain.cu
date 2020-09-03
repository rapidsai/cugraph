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

#include <community/louvain.cuh>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<int, weight_t> louvain(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                                 vertex_t *louvain_parts,
                                 int max_level,
                                 weight_t resolution,
                                 cudaStream_t stream)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, louvain expects a weighted graph");
  CUGRAPH_EXPECTS(louvain_parts != nullptr, "API error, louvain_parts is null");

  Louvain<GraphCSRView<vertex_t, edge_t, weight_t>> runner(graph, stream);

  return runner.compute(louvain_parts, max_level, resolution);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<int, weight_t> louvain(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                                 vertex_t *louvain_parts,
                                 int max_level,
                                 weight_t resolution)
{
  cudaStream_t stream{0};

  return detail::louvain(graph, louvain_parts, max_level, resolution, stream);
}

template std::pair<int, float> louvain(
  GraphCSRView<int32_t, int32_t, float> const &, int32_t *, int, float);
template std::pair<int, double> louvain(
  GraphCSRView<int32_t, int32_t, double> const &, int32_t *, int, double);

}  // namespace cugraph

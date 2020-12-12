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

#include "tsp.hpp"

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
TSP<vertex_t, edge_t, weight_t>::TSP(const raft::handle_t &handle,
    GraphCOOView<vertext_t, edge_t, weight_t> &graph,

    const float *x_pos,
    const float *y_pos,
    const int restarts)
: handle_(handle),
  x_pos_(x_pos),
  y_pos_(y_pos),
  restarts_(restarts) {

	stream_ = handle_.get_stream();
  sort(graph, stream_);
  src_ = graph.src_indices;
  dst_ = graph.dst_indices;
  weights_ = graph.weights;
  n_ = graph.number_of_vertices;
  e_ = graph.number_of_edges;
  max_blocks_ = handle_.get_device_properties().maxGridSize[0];
  max_threads_ = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count_ = handle_.get_device_properties().multiProcessorCount;
}

template <typename vertex_t, typename edge_t, typename weight_t>
float TSP<vertex_t, edge_t, weight_t>::compute() {
  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");

}

template <typename vertex_t, typename edge_t, typename weight_t>
float traveling_salesman(const raft::handle_t &handle,
          GraphCOOview<vertext_t, edge_t, weight_t> &graph,
          const float *x_pos,
          const float *y_pos,
          const int restarts) {

  TSP<vertex_t, edge_t, weight_t> tsp(
      handle,
      graph
      x_pos,
      y_pos,
      restarts);
  return tsp.compute();
}

template float traveling_salesman<int, int, float>(
    const raft::handle_t &handle,
    const Graph_COO<int, int, float> &graph,
    float *x_pos,
    float *y_pos,
    int restarts);

template float traveling_salesman<int, int, double>(
    const raft::handle_t &handle,
    const Graph_COO<int, int, double> &graph,
    float *x_pos,
    float *y_pos,
    int restarts);
} // namespace cugraph


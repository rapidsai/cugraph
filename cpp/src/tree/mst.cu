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

/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for MST
 *
 * @file mst.cu
 * ---------------------------------------------------------------------------**/

#include <algorithms.hpp>
#include <memory>
#include <utility>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <ctime>

#include <graph.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/mst/mst.cuh>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<GraphCOO<vertex_t, edge_t, weight_t>> mst_impl(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t *colors,
  rmm::mr::device_memory_resource *mr)

{
  RAFT_EXPECTS(colors != nullptr, "API error, must specify valid location for colors");

  auto stream    = handle.get_stream();
  auto mst_edges = raft::mst::mst<vertex_t, edge_t, weight_t>(handle,
                                                              graph.offsets,
                                                              graph.indices,
                                                              graph.edge_data,
                                                              graph.number_of_vertices,
                                                              graph.number_of_edges,
                                                              colors,
                                                              stream);

  auto out_graph = std::make_unique<GraphCOO<vertex_t, edge_t, weight_t>>(
    graph.number_of_vertices, mst_edges.n_edges, true, stream, mr);

  auto src_ptr     = out_graph->src_indices();
  auto dst_ptr     = out_graph->dst_indices();
  auto weights_ptr = out_graph->edge_data();
  src_ptr          = std::move(mst_edges.src.data());
  dst_ptr          = std::move(mst_edges.dst.data());
  weights_ptr      = std::move(mst_edges.weights.data());

  return out_graph;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<GraphCOO<vertex_t, edge_t, weight_t>> mst(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t *colors,
  rmm::mr::device_memory_resource *mr)
{
  return detail::mst_impl(handle, graph, colors, mr);
}

template std::unique_ptr<GraphCOO<int, int, float>> mst<int, int, float>(
  raft::handle_t const &handle,
  GraphCSRView<int, int, float> const &graph,
  int *colors,
  rmm::mr::device_memory_resource *mr);
template std::unique_ptr<GraphCOO<int, int, double>> mst<int, int, double>(
  raft::handle_t const &handle,
  GraphCSRView<int, int, double> const &graph,
  int *colors,
  rmm::mr::device_memory_resource *mr);
}  // namespace cugraph

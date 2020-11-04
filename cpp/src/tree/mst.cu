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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <ctime>

#include <graph.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/mst/mst.cuh>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
template std::unique_ptr<GraphCOO<vertex_t, edge_t, weight_t>> mst_impl(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t *colors,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())

{
  RAFT_EXPECTS(mst_flag != nullptr, "API error, must specify valid location for mst flag");

  auto stream  = handle.get_stream();
  auto exec    = rmm::exec_policy(stream);
  auto t_exe_p = exec->on(stream);

  auto mst_edges = raft::mst::mst<vertex_t, edge_t, weight_t>(handle,
                                                              graph.offsets,
                                                              graph.indices,
                                                              graph.weights,
                                                              graph.number_of_vertices,
                                                              graph.number_of_edges,
                                                              colors);

  auto out_graph = std::make_unique<GraphCOO<vertex_t, edge_t, weight_t>>(
    graph.number_of_vertices, mst_edges.size, true, stream, mr);
  out_graph->src_indices() = std::move(mst_edges.src);
  out_graph->dst_indices() = std::move(mst_edges.dst);
  out_graph->weights()     = std::move(mst_edges.weights);

  return out_graph;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
template std::unique_ptr<GraphCOO<vertex_t, edge_t, weight_t>> mst(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t *colors,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
{
  return detail::mst_impl(handle, graph, mst_flag);
}

template std::unique_ptr<GraphCOO<int, int, float>> mst<int, int, float>(
  raft::handle_t const &handle,
  GraphCSRView<int, int, float> const &graph,
  int *colors,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());
template std::unique_ptr<GraphCOO<int, int, double>> mst<int, int, double>(
  raft::handle_t const &handle,
  GraphCSRView<int, int, double> const &graph,
  int *colors,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());
}  // namespace cugraph

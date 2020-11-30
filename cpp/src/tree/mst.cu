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
  rmm::mr::device_memory_resource *mr)

{
  auto stream = handle.get_stream();
  rmm::device_uvector<vertex_t> colors(graph.number_of_vertices, stream);
  auto mst_edges = raft::mst::mst<vertex_t, edge_t, weight_t>(handle,
                                                              graph.offsets,
                                                              graph.indices,
                                                              graph.edge_data,
                                                              graph.number_of_vertices,
                                                              graph.number_of_edges,
                                                              colors.data(),
                                                              stream);

  GraphCOOContents<vertex_t, edge_t, weight_t> coo_contents{
    graph.number_of_vertices,
    mst_edges.n_edges,
    std::make_unique<rmm::device_buffer>(mst_edges.src.release()),
    std::make_unique<rmm::device_buffer>(mst_edges.dst.release()),
    std::make_unique<rmm::device_buffer>(mst_edges.weights.release())};

  return std::make_unique<GraphCOO<vertex_t, edge_t, weight_t>>(std::move(coo_contents));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<GraphCOO<vertex_t, edge_t, weight_t>> minimum_spanning_tree(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::mr::device_memory_resource *mr)
{
  return detail::mst_impl(handle, graph, mr);
}

template std::unique_ptr<GraphCOO<int, int, float>> minimum_spanning_tree<int, int, float>(
  raft::handle_t const &handle,
  GraphCSRView<int, int, float> const &graph,
  rmm::mr::device_memory_resource *mr);
template std::unique_ptr<GraphCOO<int, int, double>> minimum_spanning_tree<int, int, double>(
  raft::handle_t const &handle,
  GraphCSRView<int, int, double> const &graph,
  rmm::mr::device_memory_resource *mr);
}  // namespace cugraph

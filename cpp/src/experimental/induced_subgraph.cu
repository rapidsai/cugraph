/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <utilities/device_comm.cuh>
#include <utilities/error.hpp>
#include <utilities/host_scalar_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cuco/static_map.cuh>

#include <tuple>

namespace cugraph {
namespace experimental {
namespace detail {
}  // detail

/**
 * @brief extract induced subgraph(s). 
 * 
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed 
 * @tparam store_transposed Flag indicating whether to store the graph adjacency matrix as is or as
 * transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of, we extract induced subgraphs from @p graph_view.
 * @param subgraph_offsets Pointer to subgraph vertex offsets (size == @p num_subgraphs + 1).
 * @param subgraph_vertices Pointer to subgraph vertices (size == @p subgraph_offsets[@p num_subgraphs]).
 * @param num_subgraphs Number of induced subgraphs to extract.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>, rmm::device_uvector<size_t>> Quadraplet of edge source vertices, edge destination vertices, edge weights, and edge offsets for each induced subgraph.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>, rmm::device_uvector<size_t>>
extract_induced_subgraph(
  raft::handle_t const &handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transpsoed, multi_gpu> const &graph_view,
  size_t const * subgraph_offsets/* size == num_subgraphs + 1 */,
  vertex_t const * subgraph_vertices/* size == subgraph_offsets[num_subgraphs] */,
  size_t num_subgraphs,
  bool do_expensive_check = false) {
  // FIXME: this code is inefficient for the vertices with their local degrees much larger than the number of vertices in the subgraphs. We may later add additional code to handle such cases.
  // FIXME: we may consider the performance (speed & memory footprint, hash based approach uses extra-memory) of hash table based and binary search based approaches

  rmm::device_uvector<size_t> subgraph_major_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> subgraph_majors(0, handle.get_stream());
  rmm::device_uvector<size_t> subgraph_minor_offsets(0, handle.get_stream());  // relevant only if multi_gpu
  rmm::device_uvector<vertex_t> subgraph_minors(0, handle.get_stream());  // relevant only if multi_gpu

  // 1. construct (subgraph_idx, vertex, local_degree) triplets

  std::vector<size_t> h_subgraph_offsets(num_subgraphs + 1);
  raft::update_host(h_subgraph_offsets.data(), subgraph_offsets, num_subgraphs + 1, handle.get_stream());

  // construct (subgraph_idx, v, local_degree)

  // sort triplets by local_degree (non-ascending)

  // find number of edges for each subgraph

  // allocate memory

  // enumerate edges for each subgraph
}

}  // namespace experimental
}  // namespace cugraph

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

// Alex Fender afender@nvida.com
#include <algorithms.hpp>
#include <memory>
#include <utility>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <ctime>

#include <graph.hpp>
#include <utilities/error.hpp>
namespace {
/*
Description
Let the egonet graph of a node x be the subgraph that includes the neighborhood of x and all edges
between them. Naive algorithm
- Add center node x to the graph.
- Go through all the neighbors y of this center node x, add edge (x, y) to the graph.
- For each neighbor y of center node x, go through all the neighbors z of center node x, if there is
an edge between y and z in original graph, add edge (y, z) to our new graph.

Rather than doing custom one/two hops features, we propose a generic k-hops solution leveraging BFS
cutoff and subgraph extraction
*/

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<GraphCOO<vertex_t, edge_t, weight_t>> extract_ego(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &csr_view,
  vertex_t source_vertex,
  vertex_t radius)
{
  auto v      = csr_view.number_of_vertices;
  auto e      = csr_view.number_of_edges;
  auto stream = handle.get_stream();

  // BFS with cutoff
  rmm::device_uvector<vertex_t> distances(v, stream);
  rmm::device_uvector<vertex_t> predecessors(v, stream);
  bool direction_optimizing = false;
  cugraph::experimental::bfs(
    handle, csr_view, distances, predecessors, source_vertex, direction_optimizing, radius);

  // identify reached vertices from disatnce array
  rmm::device_uvector<vertex_t> v_id(v, stream);
  rmm::device_uvector<vertex_t> reached(v, stream);
  thrust::sequence(rmm::exec_policy(stream)->on(stream), v_id.begin(), v_id.end(), 0);
  auto reached_end =
    thrust::remove_copy_if(rmm::exec_policy(stream)->on(stream),
                           vid.begin(),
                           vid.end(),
                           distances,
                           reached.begin(),
                           thrust::equal_to<vertex_t>(std::numeric_limits<vertex_t>::max()));

  // to COO
  rmm::device_uvector<vertex_t> d_src(e, stream);
  offsets_to_indices(csr_view.offsets, v, d_src);
  GraphCOOView<vertex_t, edge_t, weight_t> coo_view;
  coo_view.number_of_vertices = v;
  coo_view.number_of_edges    = e;
  coo_view.src_indices        = d_src;
  coo_view.dst_indices        = csr_view.indices;
  coo_view.edge_data          = csr_view.edge_data;

  // extract
  auto n_reached = thrust::distance(reached.begin(), reached_end);
  return extract_subgraph_vertex(coo_view, reached.data().get(), n_reached);
}
}  // namespace

namespace cugraph {
namespace egonet {
template std::unique_ptr<GraphCOO<int32_t, int32_t, float>> extract<int32_t, int32_t, float>(
  raft::handle_t const &, GraphCSRView<int32_t, int32_t, float> const &, int32_t, int32_t);
}  // namespace egonet
}  // namespace cugraph
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

// Alex Fender afender@nvida.com
#include <algorithms.hpp>
#include <memory>
#include <tuple>
#include <utility>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <ctime>

#include <graph.hpp>

#include <utilities/error.hpp>
#include "experimental/graph.hpp"
#include "utilities/graph_utils.cuh"

#include <experimental/graph_view.hpp>

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
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<size_t>>
extract(
  raft::handle_t const &handle,
  cugraph::experimental::graph_view_t<vertex_t, edge_t, weight_t, false, false> const &csr_view,
  vertex_t *source_vertex,
  vertex_t n_subgraphs,
  vertex_t radius)
{
  auto v      = csr_view.get_number_of_vertices();
  auto e      = csr_view.get_number_of_edges();
  auto stream = handle.get_stream();

  // BFS with cutoff
  rmm::device_vector<vertex_t> reached(v);
  rmm::device_vector<vertex_t> predecessors(v);  // not used
  bool direction_optimizing = false;

  cugraph::experimental::bfs<vertex_t, edge_t, weight_t, false>(handle,
                                                                csr_view,
                                                                reached.data().get(),
                                                                predecessors.data().get(),
                                                                source_vertex[0],
                                                                direction_optimizing,
                                                                radius);

  // identify reached vertex ids from distance array
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator(vertex_t{0}),
                    thrust::make_counting_iterator(v),
                    reached.begin(),
                    reached.begin(),
                    [sentinel = std::numeric_limits<vertex_t>::max()] __device__(
                      auto id, auto val) { return val < sentinel ? id : sentinel; });
  // removes unreached data
  auto reached_end = thrust::remove(rmm::exec_policy(stream)->on(stream),
                                    reached.begin(),
                                    reached.end(),
                                    std::numeric_limits<vertex_t>::max());

  // extract
  vertex_t n_reached = thrust::distance(reached.begin(), reached_end);

  // return cugraph::experimental::extract_subgraph_vertex(csr_view, v, reached.data().get(),
  // n_reached);

  rmm::device_uvector<vertex_t> src_indices(v, stream);
  rmm::device_uvector<vertex_t> dst_indices(v, stream);
  rmm::device_uvector<weight_t> weights(v, stream);
  rmm::device_uvector<size_t> off(v, stream);

  return std::make_tuple(
    std::move(src_indices), std::move(dst_indices), std::move(weights), std::move(off));
}
}  // namespace
namespace cugraph {
namespace experimental {
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &handle,
            graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
            vertex_t *source_vertex,
            vertex_t n_subgraphs,
            vertex_t radius)
{
  CUGRAPH_EXPECTS(n_subgraphs == 1, "concurrent egonet subgraphs are not supported yet");
  CUGRAPH_EXPECTS(radius > 1, "radius should be at least 1");
  CUGRAPH_EXPECTS(radius < graph_view.get_number_of_vertices(), "radius is too large");

  return extract<vertex_t, edge_t, weight_t>(
    handle, graph_view, source_vertex, n_subgraphs, radius);
}

// SG FP32
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &,
            graph_view_t<int32_t, int32_t, float, false, false> const &,
            int32_t *,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &,
            graph_view_t<int32_t, int64_t, float, false, false> const &,
            int32_t *,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &,
            graph_view_t<int64_t, int64_t, float, false, false> const &,
            int64_t *,
            int64_t,
            int64_t);

// SG FP64
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &,
            graph_view_t<int32_t, int32_t, double, false, false> const &,
            int32_t *,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &,
            graph_view_t<int32_t, int64_t, double, false, false> const &,
            int32_t *,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const &,
            graph_view_t<int64_t, int64_t, double, false, false> const &,
            int64_t *,
            int64_t,
            int64_t);
}  // namespace experimental
}  // namespace cugraph
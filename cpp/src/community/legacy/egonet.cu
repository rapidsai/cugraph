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

// Alex Fender afender@nvida.com
#include <cstddef>
#include <cugraph/algorithms.hpp>
#include <memory>
#include <tuple>
#include <utility>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <ctime>
#include <thrust/transform.h>

#include <cugraph/legacy/graph.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/utilities/error.hpp>
#include <utilities/graph_utils.cuh>

#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <utilities/high_res_timer.hpp>

namespace {

/*
Description
Let the egonet graph of a node x be the subgraph that includes node x, the neighborhood of x, and
all edges between them. Naive algorithm
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
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract(raft::handle_t const& handle,
        cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, false> const& csr_view,
        vertex_t* source_vertex,
        vertex_t n_subgraphs,
        vertex_t radius)
{
  auto v                = csr_view.number_of_vertices();
  auto user_stream_view = handle.get_stream();
  rmm::device_vector<size_t> neighbors_offsets(n_subgraphs + 1);
  rmm::device_vector<vertex_t> neighbors;

  std::vector<size_t> h_neighbors_offsets(n_subgraphs + 1);

  // Streams will allocate concurrently later
  std::vector<rmm::device_uvector<vertex_t>> reached{};
  reached.reserve(n_subgraphs);
  for (vertex_t i = 0; i < n_subgraphs; i++) {
    // Allocations and operations are attached to the worker stream
    rmm::device_uvector<vertex_t> local_reach(v, handle.get_next_usable_stream(i));
    reached.push_back(std::move(local_reach));
  }

  user_stream_view.synchronize();
#ifdef TIMING
  HighResTimer hr_timer;
  hr_timer.start("ego_neighbors");
#endif

  for (vertex_t i = 0; i < n_subgraphs; i++) {
    // get light handle from worker pool
    raft::handle_t light_handle(handle.get_next_usable_stream(i));
    auto worker_stream_view = light_handle.get_stream();

    // BFS with cutoff
    // consider adding a device API to BFS (ie. accept source on the device)
    rmm::device_uvector<vertex_t> predecessors(v, worker_stream_view);  // not used
    bool direction_optimizing = false;
    thrust::fill(rmm::exec_policy(worker_stream_view),
                 reached[i].begin(),
                 reached[i].end(),
                 std::numeric_limits<vertex_t>::max());
    thrust::fill(
      rmm::exec_policy(worker_stream_view), reached[i].begin(), reached[i].begin() + 100, 1.0);

    cugraph::bfs<vertex_t, edge_t, weight_t, false>(light_handle,
                                                    csr_view,
                                                    reached[i].data(),
                                                    predecessors.data(),
                                                    source_vertex + i,
                                                    1,
                                                    direction_optimizing,
                                                    radius);

    // identify reached vertex ids from distance array
    thrust::transform(rmm::exec_policy(worker_stream_view),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(v),
                      reached[i].begin(),
                      reached[i].begin(),
                      [sentinel = std::numeric_limits<vertex_t>::max()] __device__(
                        auto id, auto val) { return val < sentinel ? id : sentinel; });

    // removes unreached data
    auto reached_end = thrust::remove(rmm::exec_policy(worker_stream_view),
                                      reached[i].begin(),
                                      reached[i].end(),
                                      std::numeric_limits<vertex_t>::max());
    // release temp storage
    reached[i].resize(thrust::distance(reached[i].begin(), reached_end), worker_stream_view);
    reached[i].shrink_to_fit(worker_stream_view);
  }

  // wait on every one to identify their neighboors before proceeding to concatenation
  handle.sync_stream_pool();

  // Construct neighboors offsets (just a scan on neighborhod vector sizes)
  h_neighbors_offsets[0] = 0;
  for (vertex_t i = 0; i < n_subgraphs; i++) {
    h_neighbors_offsets[i + 1] = h_neighbors_offsets[i] + reached[i].size();
  }
  raft::update_device(neighbors_offsets.data().get(),
                      &h_neighbors_offsets[0],
                      n_subgraphs + 1,
                      user_stream_view.value());
  neighbors.resize(h_neighbors_offsets[n_subgraphs]);
  user_stream_view.synchronize();

  // Construct the neighboors list concurrently
  for (vertex_t i = 0; i < n_subgraphs; i++) {
    auto worker_stream_view = handle.get_next_usable_stream(i);
    thrust::copy(rmm::exec_policy(worker_stream_view),
                 reached[i].begin(),
                 reached[i].end(),
                 neighbors.begin() + h_neighbors_offsets[i]);

    // reached info is not needed anymore
    reached[i].resize(0, worker_stream_view);
    reached[i].shrink_to_fit(worker_stream_view);
  }

  // wait on every one before proceeding to grouped extraction
  handle.sync_stream_pool();

#ifdef TIMING
  hr_timer.stop();
  hr_timer.display(std::cout);
#endif

  // extract
  return cugraph::extract_induced_subgraphs(
    handle, csr_view, neighbors_offsets.data().get(), neighbors.data().get(), n_subgraphs);
}

}  // namespace

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
            vertex_t* source_vertex,
            vertex_t n_subgraphs,
            vertex_t radius)
{
  if (multi_gpu) {
    CUGRAPH_FAIL("Unimplemented.");
    return std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                           rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                           rmm::device_uvector<weight_t>(0, handle.get_stream()),
                           rmm::device_uvector<size_t>(0, handle.get_stream()));
  }
  CUGRAPH_EXPECTS(n_subgraphs > 0, "Need at least one source to extract the egonet from");
  CUGRAPH_EXPECTS(n_subgraphs < graph_view.number_of_vertices(),
                  "Can't have more sources to extract from than vertices in the graph");
  CUGRAPH_EXPECTS(radius > 0, "Radius should be at least 1");
  CUGRAPH_EXPECTS(radius < graph_view.number_of_vertices(), "radius is too large");
  // source_vertex range is checked in bfs.

  return extract<vertex_t, edge_t, weight_t>(
    handle, graph_view, source_vertex, n_subgraphs, radius);
}

// SG FP32
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int32_t, int32_t, float, false, false> const&,
            int32_t*,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int32_t, int64_t, float, false, false> const&,
            int32_t*,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int64_t, int64_t, float, false, false> const&,
            int64_t*,
            int64_t,
            int64_t);

// SG FP64
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int32_t, int32_t, double, false, false> const&,
            int32_t*,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int32_t, int64_t, double, false, false> const&,
            int32_t*,
            int32_t,
            int32_t);
template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int64_t, int64_t, double, false, false> const&,
            int64_t*,
            int64_t,
            int64_t);
}  // namespace cugraph

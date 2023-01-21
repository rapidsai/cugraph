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
#pragma once

//#define TIMING

#include <utilities/graph_utils.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#ifdef TIMING
#include <cugraph/utilities/high_res_timer.hpp>
#endif

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <ctime>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform.h>

#include <cstddef>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

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

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract(raft::handle_t const& handle,
        cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
        std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
        raft::device_span<vertex_t const> source_vertex,
        vertex_t radius,
        bool do_expensive_check)
{
  auto user_stream_view = handle.get_stream();

  size_t num_sources = source_vertex.size();
  [[maybe_unused]] std::vector<size_t> source_start{{0}};

  if constexpr (multi_gpu) {
    source_start =
      cugraph::host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());
    num_sources = std::reduce(source_start.begin(), source_start.end());
    std::exclusive_scan(source_start.begin(), source_start.end(), source_start.begin(), size_t{0});
  }

  rmm::device_uvector<size_t> neighbors_offsets(num_sources + 1, user_stream_view);
  rmm::device_uvector<vertex_t> neighbors(0, user_stream_view);

  std::vector<size_t> h_neighbors_offsets(num_sources + 1);

  // Streams will allocate concurrently later
  std::vector<rmm::device_uvector<vertex_t>> reached{};
  reached.reserve(num_sources);

  user_stream_view.synchronize();
#ifdef TIMING
  HighResTimer hr_timer;
  hr_timer.start("ego_neighbors");
#endif

  // FIXME: Explore the performance here.  Single-seed BFS
  // has a slow parallelism ramp up, would we be better off
  // using the technique in induced subgraph where we tag
  // the vertices and search for matches until the frontiers
  // are large enough to use this approach?

  for (size_t i = 0; i < num_sources; i++) {
    // get light handle from worker pool
    raft::handle_t light_handle(handle.get_next_usable_stream(i));
    auto worker_stream_view = multi_gpu ? handle.get_stream() : light_handle.get_stream();

    rmm::device_uvector<vertex_t> local_reach(graph_view.local_vertex_partition_range_size(),
                                              worker_stream_view);
    reached.push_back(std::move(local_reach));

    // BFS with cutoff
    // consider adding a device API to BFS (ie. accept source on the device)
    bool direction_optimizing = false;
    thrust::fill(rmm::exec_policy(worker_stream_view),
                 reached[i].begin(),
                 reached[i].end(),
                 std::numeric_limits<vertex_t>::max());

    raft::device_span<vertex_t const> source{source_vertex.data() + i, 1};

    if constexpr (multi_gpu) {
      auto it  = std::upper_bound(source_start.begin(), source_start.end(), i);
      auto gpu = thrust::distance(source_start.begin(), it) - 1;

      if (gpu == handle.get_comms().get_rank())
        source = raft::device_span<vertex_t const>{source_vertex.data() + i - source_start[gpu], 1};
      else
        source = raft::device_span<vertex_t const>{source_vertex.data(), size_t{0}};
    }

    cugraph::bfs<vertex_t, edge_t, multi_gpu>(multi_gpu ? handle : light_handle,
                                              graph_view,
                                              reached[i].data(),
                                              nullptr,
                                              source.data(),
                                              source.size(),
                                              direction_optimizing,
                                              radius,
                                              do_expensive_check);

    // identify reached vertex ids from distance array
    thrust::transform(
      rmm::exec_policy(worker_stream_view),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      reached[i].begin(),
      reached[i].begin(),
      [sentinel = std::numeric_limits<vertex_t>::max()] __device__(auto id, auto val) {
        return val < sentinel ? id : sentinel;
      });

    // removes unreached data
    auto reached_end = thrust::remove(rmm::exec_policy(worker_stream_view),
                                      reached[i].begin(),
                                      reached[i].end(),
                                      std::numeric_limits<vertex_t>::max());

    // release temp storage
    reached[i].resize(thrust::distance(reached[i].begin(), reached_end), worker_stream_view);
    reached[i].shrink_to_fit(worker_stream_view);
  }

  // wait on every one to identify their neighbors before proceeding to concatenation
  handle.sync_stream_pool();

  // Construct neighbors offsets (just a scan on neighborhod vector sizes)
  h_neighbors_offsets[0] = 0;
  for (size_t i = 0; i < num_sources; i++) {
    h_neighbors_offsets[i + 1] = h_neighbors_offsets[i] + reached[i].size();
  }
  raft::update_device(
    neighbors_offsets.data(), &h_neighbors_offsets[0], num_sources + 1, user_stream_view.value());
  neighbors.resize(h_neighbors_offsets[num_sources], user_stream_view.value());
  user_stream_view.synchronize();

  // Construct the neighbors list concurrently
  for (size_t i = 0; i < num_sources; i++) {
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
  hr_timer.display_and_clear(std::cout);
#endif

  // extract
  return cugraph::extract_induced_subgraphs(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<size_t const>(neighbors_offsets.data(), neighbors_offsets.size()),
    raft::device_span<vertex_t const>(neighbors.data(), neighbors.size()),
    do_expensive_check);
}

}  // namespace

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            vertex_t* source_vertex,
            vertex_t n_subgraphs,
            vertex_t radius)
{
  CUGRAPH_EXPECTS(n_subgraphs > 0, "Need at least one source to extract the egonet from");
  CUGRAPH_EXPECTS(n_subgraphs < graph_view.number_of_vertices(),
                  "Can't have more sources to extract from than vertices in the graph");
  CUGRAPH_EXPECTS(radius > 0, "Radius should be at least 1");
  CUGRAPH_EXPECTS(radius < graph_view.number_of_vertices(), "radius is too large");
  // source_vertex range is checked in bfs.

  return extract(handle,
                 graph_view,
                 edge_weight_view,
                 raft::device_span<vertex_t const>{source_vertex, static_cast<size_t>(n_subgraphs)},
                 radius,
                 false);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            raft::device_span<vertex_t const> source_vertex,
            vertex_t radius,
            bool do_expensive_check)
{
  CUGRAPH_EXPECTS(radius > 0, "Radius should be at least 1");
  CUGRAPH_EXPECTS(radius < graph_view.number_of_vertices(), "radius is too large");

  return extract(handle, graph_view, edge_weight_view, source_vertex, radius, do_expensive_check);
}

}  // namespace cugraph

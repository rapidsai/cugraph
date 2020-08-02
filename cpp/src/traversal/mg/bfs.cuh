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

#pragma once

#include <raft/handle.hpp>
#include "common_utils.cuh"
#include "load_balance.cuh"
#include "../traversal_common.cuh"
#include <utilities/high_res_timer.hpp>

#include <utilities/high_res_timer.hpp>

namespace cugraph {

namespace mg {

template <typename vertex_t>
__global__
void fill_kernel(
         vertex_t *distances,
         vertex_t count,
         vertex_t start_vertex) {
  vertex_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid >= count) {
    return;
  }
  if (tid == start_vertex) {
    distances[tid] = vertex_t{0};
  } else {
    distances[tid] = cugraph::detail::traversal::vec_t<vertex_t>::max;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void fill_max_dist(raft::handle_t const &handle,
         cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
         vertex_t start_vertex,
         vertex_t *distances) {
  vertex_t array_size = graph.number_of_vertices;
  vertex_t threads = 256;
  vertex_t blocks = raft::div_rounding_up_safe(array_size, threads);
  fill_kernel<<<blocks, threads, 0, handle.get_stream()>>>(
      distances, array_size, start_vertex);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void bfs(raft::handle_t const &handle,
         cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
         vertex_t *distances,
         vertex_t *predecessors,
         const vertex_t start_vertex)
{
  HighResTimer timer;
  HighResTimer main_loop_timer;
  CUGRAPH_EXPECTS(handle.comms_initialized(),
                  "cugraph::mg::bfs() expected to work only in multi gpu case.");

  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  rmm::device_vector<unsigned> isolated_bmap(word_count, 0);
  rmm::device_vector<unsigned> visited_bmap(word_count, 0);
  rmm::device_vector<unsigned> output_frontier_bmap(word_count, 0);

  rmm::device_vector<unsigned> unique_bmap(word_count, 0);

  //Buffers required for BFS
  rmm::device_vector<vertex_t> input_frontier(graph.number_of_vertices);
  rmm::device_vector<vertex_t> output_frontier(graph.number_of_vertices);
  rmm::device_vector<size_t> temp_buffer_len(handle.get_comms().get_size());

  // Load balancer for calls to bfs functors
  detail::LoadBalanceExecution<vertex_t, edge_t, weight_t> lb(handle, graph);

  cudaStream_t stream = handle.get_stream();

  timer.start("create_isolated_bitmap");
  //Reusing buffers to create isolated bitmap
  {
    rmm::device_vector<vertex_t>& local_isolated_ids = input_frontier;
    rmm::device_vector<vertex_t>& global_isolated_ids = output_frontier;
    detail::create_isolated_bitmap(
        handle, graph,
        local_isolated_ids, global_isolated_ids,
        temp_buffer_len, isolated_bmap);
  }
  timer.stop();

  //TODO : Check if starting vertex is isolated. Exit function if it is.

  //Initialize input frontier
  input_frontier[0] = start_vertex;
  vertex_t input_frontier_len = 1;

  timer.start("fill_max_dist");
  //Start at level 0
  vertex_t level = 0;
  if (distances != nullptr) {
    fill_max_dist(handle, graph, start_vertex, distances);
  }
  timer.stop();

  timer.start("fill invalid vertex id");
  // Fill predecessors with invalid vertex id
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               predecessors,
               predecessors + graph.number_of_vertices,
               cugraph::invalid_idx<vertex_t>::value);
  timer.stop();

  do {
    main_loop_timer.start("main_loop");
    timer.start("add_to_bitmap");
    //Mark all input frontier vertices as visited
    detail::add_to_bitmap(handle, visited_bmap, input_frontier, input_frontier_len);
    timer.stop();

    timer.start("clear output frontier bitmap");
    //Clear output frontier bitmap
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 output_frontier_bmap.begin(),
                 output_frontier_bmap.end(),
                 static_cast<unsigned>(0));
    timer.stop();

    ++level;

    //output_frontier.resize(graph.number_of_vertices);
    vertex_t output_frontier_len = 0;
    cudaStreamSynchronize(stream);
    //Generate output frontier bitmap from input frontier
    if (distances != nullptr) {
      timer.start("create functor");
      // BFS Functor for frontier calculation
      detail::bfs_pred_dist<vertex_t, edge_t> bfs_op(
        output_frontier_bmap.data().get(),
        isolated_bmap.data().get(),
        visited_bmap.data().get(),
        predecessors, distances, level);
      timer.stop();
      timer.start("RUN functor");
      output_frontier_len =
        lb.run(bfs_op, input_frontier, input_frontier_len, output_frontier);
      timer.stop();
    } else {
      timer.start("create functor");
      // BFS Functor for frontier calculation
      detail::bfs_pred<vertex_t, edge_t> bfs_op(
        output_frontier_bmap.data().get(),
        isolated_bmap.data().get(),
        visited_bmap.data().get(),
        predecessors);
      timer.stop();
      timer.start("RUN functor");
      output_frontier_len =
        lb.run(bfs_op, input_frontier, input_frontier_len, output_frontier);
      timer.stop();
    }

    timer.start("collect_frontier");
    input_frontier_len =
      detail::collect_frontier(
          handle, graph,
          temp_buffer_len,
          unique_bmap,
          output_frontier,
          output_frontier_len,
          input_frontier);
    timer.stop();

    main_loop_timer.stop();
  } while (input_frontier_len != 0);

  timer.start("collect predecessors");
  // In place reduce to collect predecessors
  if (handle.comms_initialized()) {
    handle.get_comms().allreduce(predecessors,
                                 predecessors,
                                 graph.number_of_vertices,
                                 raft::comms::op_t::MIN,
                                 handle.get_stream());
  }
  timer.stop();

  timer.start("collect predecessors");
  if (distances != nullptr) {
    // In place reduce to collect predecessors
    if (handle.comms_initialized()) {
      handle.get_comms().allreduce(distances,
                                   distances,
                                   graph.number_of_vertices,
                                   raft::comms::op_t::MIN,
                                   handle.get_stream());
    }
  }
  timer.stop();

  timer.display(std::cout);
  main_loop_timer.display(std::cout);
}

}  // namespace mg

}  // namespace cugraph

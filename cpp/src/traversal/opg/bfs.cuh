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
#include "load_balance.cuh"
#include "bfs_comms.cuh"
#include "common_utils.cuh"

namespace cugraph {

namespace opg {

template <typename VT, typename ET, typename WT>
void bfs(raft::handle_t const &handle,
    cugraph::GraphCSRView<VT, ET, WT> const &graph,
    VT *distances,
    VT *predecessors,
    const VT start_vertex) {

  using namespace detail;

  //We need to keep track if a vertex is visited or its status
  //This needs to be done for all the vertices in the global graph
  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  rmm::device_vector<unsigned> input_frontier(word_count);
  rmm::device_vector<unsigned> output_frontier(word_count);
  rmm::device_vector<unsigned> visited(word_count);

  rmm::device_vector<unsigned> frontier_not_empty(1);

  //Load balancer for calls to bfs functors
  LoadBalanceExecution<VT, ET, WT> lb(handle, graph);

  //Functor to check if frontier is empty
  is_not_equal neq(static_cast<unsigned>(0), frontier_not_empty.data().get());

  cudaStream_t stream = handle.get_stream();

  //Fill predecessors with an invalid vertex id
  thrust::fill(rmm::exec_policy(stream)->on(stream),
      predecessors, predecessors + graph.number_of_vertices,
      graph.number_of_vertices);

  VT level = 0;
  if (distances != nullptr) {
    thrust::fill(rmm::exec_policy(stream)->on(stream),
        distances, distances + graph.number_of_vertices,
        std::numeric_limits<VT>::max());
  }

  //BFS communications wrapper
  BFSCommunicatorIterativeBCastReduce<VT, ET, WT> bfs_comm(handle, word_count);

  //0. 'Insert' starting vertex in the input frontier
  input_frontier[start_vertex/BitsPWrd<unsigned>] =
    static_cast<unsigned>(1)<<(start_vertex%BitsPWrd<unsigned>);

  do {
    //1. Mark all input frontier vertices as visited
    thrust::transform(rmm::exec_policy(stream)->on(stream),
        input_frontier.begin(), input_frontier.end(),
        visited.begin(),
        visited.begin(),
        bitwise_or());

    //2. Clear out output frontier
    thrust::fill(
        output_frontier.begin(),
        output_frontier.end(),
        static_cast<unsigned>(0));

    //3. Create output frontier from input frontier
    if (distances != nullptr) {
      //BFS Functor for frontier calculation
      detail::bfs_frontier_pred_dist<VT> bfs_op(
          output_frontier.data().get(), predecessors, distances, level++);
      lb.run(bfs_op, input_frontier.data().get());
    } else {
      //BFS Functor for frontier calculation
      detail::bfs_frontier_pred<VT> bfs_op(
          output_frontier.data().get(), predecessors);
      lb.run(bfs_op, input_frontier.data().get());
    }

    //3a. Combine output frontier from all GPUs
    bfs_comm.allreduce(output_frontier);

    //4. 'Remove' all vertices in output frontier
    //that are already visited
    thrust::transform(rmm::exec_policy(stream)->on(stream),
        visited.begin(), visited.end(),
        output_frontier.begin(),
        output_frontier.begin(),
        remove_visited());

    //5. Use the output frontier as input for the next step
    input_frontier.swap(output_frontier);

    //6. If all bits in input frontier are inactive then bfs is done
    frontier_not_empty[0] = 0;
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
        input_frontier.begin(), input_frontier.end(),
        neq);
  } while (frontier_not_empty[0] == 1);

  //In place reduce to collect predecessors
  handle.get_comms().allreduce(
      predecessors, predecessors,
      graph.number_of_vertices,
      raft::comms::op_t::MIN,
      handle.get_stream());

  //If the bfs loop does not assign a predecessor for a vertex
  //then its value will be graph.number_of_vertices. This needs to be
  //replaced by invalid vertex id to denote that a vertex does have
  //a predecessor
  thrust::replace(rmm::exec_policy(stream)->on(stream),
      predecessors, predecessors + graph.number_of_vertices,
      graph.number_of_vertices,
      cugraph::invalid_vertex_id<VT>::value);

  if (distances != nullptr) {
    //In place reduce to collect predecessors
    handle.get_comms().allreduce(
        distances, distances,
        graph.number_of_vertices,
        raft::comms::op_t::MIN,
        handle.get_stream());
  }

}

}//namespace opg

}//namespace cugraph

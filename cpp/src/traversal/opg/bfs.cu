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

#include <raft/handle.hpp>
#include "load_balance.cuh"
#include "bfs_comms.cuh"

namespace cugraph {

namespace detail {

namespace opg {

struct bitwise_or {
  __device__ unsigned operator()(unsigned& a, unsigned & b) { return a | b; }
};

struct remove_visited {
  __device__ unsigned operator()(unsigned& visited, unsigned& output) {
    //OUTPUT AND VISITED - common bits between output and visited
    //OUTPUT AND (NOT (OUTPUT AND VISITED))
    // - remove common bits between output and visited from output
    return (output & (~( output & visited )));
  }
};

template <typename VT>
struct bfs_frontier_pred {
  unsigned * output_frontier_;
  VT * predecessors_;

  bfs_frontier_pred(
      unsigned * output_frontier,
      VT * predecessors) :
    output_frontier_(output_frontier),
    predecessors_(predecessors) {}

  __device__ void operator()(VT src, VT dst) {
    unsigned active_bit = static_cast<unsigned>(1)<<(dst % BitsPWrd<unsigned>);
    unsigned prev_word =
      atomicOr(output_frontier_ + (dst/BitsPWrd<unsigned>), active_bit);
    //If this thread activates the frontier bitmap for a destination
    //then the source is the predecessor of that destination
    if (prev_word & active_bit == 0) {
      predecessors_[dst] = src;
    }
  }
};

template <typename VT>
struct bfs_frontier_pred_dist {
  unsigned * output_frontier_;
  VT * predecessors_;
  VT * distances_;
  VT level_;

  bfs_frontier_pred_dist(
      unsigned * output_frontier,
      VT * predecessors,
      VT * distances, VT level) :
    output_frontier_(output_frontier),
    predecessors_(predecessors),
    distances_(distances),
    level_(level) {}

  __device__ void operator()(VT src, VT dst) {
    unsigned active_bit = static_cast<unsigned>(1)<<(dst % BitsPWrd<unsigned>);
    unsigned prev_word =
      atomicOr(output_frontier_ + (dst/BitsPWrd<unsigned>), active_bit);
    //If this thread activates the frontier bitmap for a destination
    //then the source is the predecessor of that destination
    if (prev_word & active_bit == 0) {
      distances_[dst] = level_;
      predecessors_[dst] = src;
    }
  }
};

struct is_not_equal {
  unsigned cmp_;
  unsigned * flag_;
  is_not_equal(unsigned cmp, unsigned * flag) : cmp_(cmp), flag_(flag) {}
  __device__ void operator()(unsigned& val) {
    if (val != cmp_) {
      *flag_ = 1;
    }
  }
};

template <typename VT, typename ET, typename WT>
void bfs(raft::handle_t const &handle,
    cugraph::experimental::GraphCSRView<VT, ET, WT>& graph,
    VT *distances,
    VT *predecessors,
    const VT start_vertex) {

  //We need to keep track if a vertex is visited or its status
  //This needs to be done for all the vertices in the global graph
  size_t word_count = number_of_words(graph.number_of_vertices);
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
  BFSCommunicator<VT, ET, WT> bfs_comm(handle, word_count);

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
      bfs_frontier_pred_dist<VT> bfs_op(
          output_frontier.data().get(), predecessors, distances, level++);
      lb.run(bfs_op, input_frontier.data().get());
    } else {
      //BFS Functor for frontier calculation
      bfs_frontier_pred<VT> bfs_op(
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
      cugraph::experimental::invalid_vertex_id<VT>::value);

  if (distances != nullptr) {
    //In place reduce to collect predecessors
    handle.get_comms().allreduce(
        distances, distances,
        graph.number_of_vertices,
        raft::comms::op_t::MIN,
        handle.get_stream());
  }

}

template void bfs(raft::handle_t const &handle,
    cugraph::experimental::GraphCSRView<int, int, float> &graph,
    int *distances,
    int *predecessors,
    const int start_vertex);

template void bfs(raft::handle_t const &handle,
    cugraph::experimental::GraphCSRView<int, int, double> &graph,
    int *distances,
    int *predecessors,
    const int start_vertex);


}//namespace opg

}//namespace detail

}//namespace cugraph

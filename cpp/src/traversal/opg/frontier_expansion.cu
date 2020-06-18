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


namespace detail {

namespace opg {

constexpr inline size_t
divup(const size_t& numerator, const size_t& denominator) {
  return (numerator + denominator - 1)/denominator;
}

size_t number_of_words(size_t number_of_bits)
{
  size_t numerator = number_of_bits;
  size_t denominator = BitsPWrd<unsigned>;
  return (numerator > denominator) ?
    1 + divup(numerator - denominator, denominator) :
    (denominator > 0);
}

struct bitwise_or {
  __device__ unsigned operator()(unsigned& a, unsigned & b) { return a | b; }
};

void aggregate_visited_frontier(
    rmm::device_vector<unsigned>& input,
    rmm::device_vector<unsigned>& visited,
    cudaStream_t stream) {
  CUGRAPH_EXPECTS(input.size() == visited.size(),
      "Size of input frontier vector and visited vector should be equal");
  thrust::transform(rmm::exec_policy(stream)->on(stream),
      input.begin(), input.end(),
      visited.begin(),
      visited.begin(),
      bitwise_or());
}

template <typename VT>
struct bitwise_atomic_or {
  unsigned * output_frontier_;
  VT * predecessors_;

  bitwise_atomic_or(
      unsigned * output_frontier,
      VT * predecessors) :
    output_frontier_(output_frontier),
    predecessors_(predecessors) {}

  __device__ void operator()(VT src, VT dst) {
    unsigned active_bit = static_cast<unsigned>(1)<<(dst % BitsPWrd<unsigned>);
    unsigned prev_word =
      atomicOr(output_frontier + (dst/BitsPWrd<unsigned>), active_bit);
    //If this thread activates the frontier bitmap for a destination
    //then the source is the predecessor of that destination
    if (prev_word & active_word == 0) {
      predecessors[dst] = src;
    }
  }
};

template <typename VT, typename ET, typename WT>
void expand_frontier(
    rmm::device_vector<unsigned>& input,
    LoadBalanceExecution<VT, ET, WT>& lb,
    rmm::device_vector<unsigned>& output,
    VT * predecessors,
    cudaStream_t stream) {
  bitwise_atomic_or<VT> bfs_op(output.data(), predecessors);
  lb.run(bfs_op, input.data(), stream);
}

template <typename VT, typename ET, typename WT>
void bfs(raft::handle_t const &handle,
    cugraph::experimental::GraphCSRView<VT, ET, WT>& graph,
    VT *predecessors,
    const VT start_vertex) {
  LoadBalanceExecution<VT, ET, WT> lb(graph);
  size_t buffer_size = number_of_words(graph.number_of_vertices);

  //We need to keep track if a vertex is visited or its status
  //This needs to be done for all the vertices in the global graph
  rmm::device_vector<unsigned> input_frontier(buffer_size);
  rmm::device_vector<unsigned> output_frontier(buffer_size);
  rmm::device_vector<unsigned> visited(buffer_size);

  frontier[start_vertex/BitsPWrd<unsigned>] =
    static_cast<unsigned>(1)<<(start_vertex%BitsPWrd<unsigned>);

  while(true) {
    aggregate_visited_frontier(input_frontier, visited);
    thrust::fill(
        output_frontier.begin(),
        output_frontier.end(),
        static_cast<unsigned>(0));
    expand_frontier(input_frontier, lb, output_frontier, predecessors, stream);
    frontier_correction(output_frontier, visited);
    input_frontier.swap(output_frontier);
  }
}

template void bfs(raft::handle_t const &handle,
    cugraph::experimental::GraphCSRView<int, int, float> &graph,
    int *predecessors,
    const int start_vertex);


}//namespace opg

}//namespace detail

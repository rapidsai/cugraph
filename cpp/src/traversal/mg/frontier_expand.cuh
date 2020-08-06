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

#include <raft/cudart_utils.h>
#include <graph.hpp>
#include <utilities/high_res_timer.hpp>
#include "../traversal_common.cuh"
#include "frontier_expand_kernel.cuh"
#include "common_utils.cuh"

namespace cugraph {

namespace mg {

namespace detail {

#define EDGES_PER_BLOCK 2048
#define THREADS 256

template <typename vertex_t, typename edge_t>
__global__ void
write_vertex_degree(
    edge_t * offsets,
    vertex_t * input_frontier,
    vertex_t total_vertex_count,
    vertex_t vertex_begin,
    edge_t * frontier_vertex_degree) {
  edge_t id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id < total_vertex_count) {
    vertex_t source_id = input_frontier[id];
    //Adjust the vertex id to point to the offset in the local
    //partition.  Not doing so would probably cause out of bounds access
    vertex_t loc = source_id - vertex_begin;
    frontier_vertex_degree[id + 1] = offsets[loc + 1] - offsets[loc];
  }
  if (id == 0) {
    frontier_vertex_degree[0] = 0;
  }
}

template <typename vertex_t, typename edge_t>
__global__ void compute_block_offsets_kernel(
    edge_t * frontier_vertex_offset,
    vertex_t frontier_vertex_offset_len,
    edge_t total_frontier_edge_count,
    edge_t * frontier_vertex_bucket_offset,
    edge_t block_count,
    edge_t edges_proc_per_block)
{
  for (edge_t bid = blockIdx.x * blockDim.x + threadIdx.x;
      bid < block_count + 1;
      bid += gridDim.x * blockDim.x) {
    edge_t eid = min(bid * edges_proc_per_block, total_frontier_edge_count);
    edge_t block_offset =
      cugraph::detail::traversal::binsearch_maxle(
          frontier_vertex_offset, eid,
          static_cast<edge_t>(0),
          frontier_vertex_offset_len - 1);
    frontier_vertex_bucket_offset[bid] = block_offset;
  }
}

template <typename vertex_t, typename edge_t>
void compute_block_offsets(
    raft::handle_t const &handle,
    rmm::device_vector<edge_t> &frontier_vertex_offset_,
    vertex_t frontier_vertex_offset_len,
    edge_t total_frontier_edge_count,
    rmm::device_vector<edge_t> &frontier_vertex_bucket_offset_,
    edge_t block_count,
    edge_t edges_proc_per_block) {
  cudaStream_t stream = handle.get_stream();
  dim3 grid, block;
  block.x = 256;
  //We need the starting and ending point of every block
  grid.x = raft::div_rounding_up_unsafe(
      block_count + 1, block.x);
  compute_block_offsets_kernel<<<grid, block, 0, stream>>>(
      frontier_vertex_offset_.data().get(),
      frontier_vertex_offset_len,
      total_frontier_edge_count,
      frontier_vertex_bucket_offset_.data().get(),
      block_count,
      edges_proc_per_block);
  CHECK_CUDA(stream);
}

template <int EdgesPerBlock, int Threads,
         typename vertex_t, typename edge_t, typename weight_t,
         typename Operator>
void
frontier_expand(
  raft::handle_t const &handle,
  cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<vertex_t> &input_frontier,
  vertex_t input_frontier_len,
  vertex_t vertex_begin,
  edge_t total_frontier_edge_count,
  rmm::device_vector<edge_t> &frontier_vertex_offset,
  rmm::device_vector<edge_t> &frontier_vertex_bucket_offset,
  edge_t blocks,
  rmm::device_vector<vertex_t> &output_frontier,
  rmm::device_vector<edge_t> &output_frontier_size,
  Operator op) {
  frontier_expand_kernel<EdgesPerBlock, Threads, EdgesPerBlock/Threads><<<
    blocks, Threads, 0, handle.get_stream()>>>(
      graph.offsets,
      graph.indices,
      input_frontier.data().get(),
      input_frontier_len,
      vertex_begin,
      total_frontier_edge_count,
      frontier_vertex_offset.data().get(),
      frontier_vertex_bucket_offset.data().get(),
      output_frontier.data().get(),
      output_frontier_size.data().get(),
      static_cast<edge_t>(output_frontier.capacity()),
      op);
}

template <typename vertex_t, typename edge_t, typename weight_t>
class FrontierOperator {
  raft::handle_t const &handle_;
  cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph_;
  vertex_t vertex_begin_;
  vertex_t vertex_end_;
  edge_t edge_count_;
  rmm::device_vector<edge_t> frontier_vertex_offset_;
  rmm::device_vector<edge_t> frontier_vertex_bucket_offset_;

  rmm::device_vector<edge_t> output_frontier_size_;
  HighResTimer timer;

public:
  FrontierOperator(
      raft::handle_t const &handle,
      cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph)
    : handle_(handle), graph_(graph)
  {
    bool is_mg = handle.comms_initialized() &&
      (graph_.local_vertices != nullptr) &&
      (graph_.local_offsets != nullptr);
    if (is_mg) {
      vertex_begin_ = graph_.local_offsets[handle_.get_comms().get_rank()];
      vertex_end_   = graph_.local_offsets[handle_.get_comms().get_rank()] +
                    graph_.local_vertices[handle_.get_comms().get_rank()];
      edge_count_ = graph_.local_edges[handle_.get_comms().get_rank()];
    } else {
      vertex_begin_ = 0;
      vertex_end_   = graph_.number_of_vertices;
      edge_count_ = graph_.number_of_edges;
    }
    frontier_vertex_offset_.resize(vertex_end_ - vertex_begin_ + 1);

    edge_t max_block_count =
      raft::div_rounding_up_unsafe(edge_count_, EDGES_PER_BLOCK);

    frontier_vertex_bucket_offset_.resize(max_block_count + 1);

    output_frontier_size_.resize(1);
  }

  template <typename Operator>
  vertex_t run(Operator op,
      rmm::device_vector<vertex_t> &input_frontier,
      vertex_t input_frontier_len,
      rmm::device_vector<vertex_t> &output_frontier)
  {
    if (input_frontier_len == 0) { return 0; }

    cudaStream_t stream = handle_.get_stream();
    //timer.start("FrontierOperator : set_frontier_degree");
    vertex_t vdeg_blocks = raft::div_rounding_up_unsafe(
        input_frontier_len, THREADS);
    write_vertex_degree<<<vdeg_blocks, THREADS, 0, stream>>>(
        graph_.offsets,
        input_frontier.data().get(),
        input_frontier_len,
        vertex_begin_,
        frontier_vertex_offset_.data().get());
    CHECK_CUDA(stream);
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
        frontier_vertex_offset_.begin() + 1,
        frontier_vertex_offset_.begin() + input_frontier_len + 1,
        frontier_vertex_offset_.begin() + 1);
    //timer.stop();
    CHECK_CUDA(stream);
    //timer.start("FrontierOperator : compute_block_offsets");
    edge_t total_frontier_edge_count =
      frontier_vertex_offset_[input_frontier_len];
    edge_t blocks_frontier_expansion = raft::div_rounding_up_unsafe(
        total_frontier_edge_count, static_cast<edge_t>(EDGES_PER_BLOCK));
    compute_block_offsets(
        handle_,
        frontier_vertex_offset_,
        input_frontier_len + 1,
        total_frontier_edge_count,
        frontier_vertex_bucket_offset_,
        blocks_frontier_expansion,
        static_cast<edge_t>(EDGES_PER_BLOCK));
    //timer.stop();

    output_frontier_size_[0] = 0;
    //timer.start("FrontierOperator : frontier_expand");
    frontier_expand<EDGES_PER_BLOCK, THREADS>(
        handle_,
        graph_,
        input_frontier,
        input_frontier_len,
        vertex_begin_,
        total_frontier_edge_count,
        frontier_vertex_offset_,
        frontier_vertex_bucket_offset_,
        blocks_frontier_expansion,
        output_frontier,
        output_frontier_size_,
        op);
    //timer.stop();

    return output_frontier_size_[0];
  }

  ~FrontierOperator(void) {
    //timer.display(std::cout);
  }
};

template <typename vertex_t>
struct BfsPredDist {
  unsigned* visited_bmap_;
  unsigned* isolated_bmap_;

  vertex_t* predecessors_;
  vertex_t* distances_;
  vertex_t level_;

  BfsPredDist(unsigned * visited_bmap,
      unsigned * isolated_bmap,
      vertex_t * predecessors,
      vertex_t * distances,
      vertex_t level) :
    visited_bmap_(visited_bmap),
    isolated_bmap_(isolated_bmap),
    predecessors_(predecessors),
    distances_(distances),
    level_(level)
  {
  }

  __device__
  bool operator()(vertex_t src, vertex_t dst)
  {
    unsigned active_bit =
      static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  =
      atomicOr(visited_bmap_ + (dst / BitsPWrd<unsigned>), active_bit);
    if (!(active_bit & prev_word)) {
      distances_[dst] = level_;
      predecessors_[dst] = src;
      bool is_dst_isolated =
        active_bit & isolated_bmap_[dst / BitsPWrd<unsigned>];
      //Indicate that dst should be pushed in queue.
      return !is_dst_isolated;
    } else {
      return false;
    }
  }
};

template <typename vertex_t>
struct BfsPred {
  unsigned* visited_bmap_;
  unsigned* isolated_bmap_;

  vertex_t* predecessors_;

  BfsPred(unsigned * visited_bmap,
      unsigned * isolated_bmap,
      vertex_t * predecessors) :
    visited_bmap_(visited_bmap),
    isolated_bmap_(isolated_bmap),
    predecessors_(predecessors)
  {
  }

  __device__
  bool operator()(vertex_t src, vertex_t dst)
  {
    unsigned active_bit =
      static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  =
      atomicOr(visited_bmap_ + (dst / BitsPWrd<unsigned>), active_bit);
    if (!(active_bit & prev_word)) {
      predecessors_[dst] = src;
      bool is_dst_isolated =
        active_bit & isolated_bmap_[dst / BitsPWrd<unsigned>];
      //Indicate that dst should be pushed in queue.
      return !is_dst_isolated;
    } else {
      return false;
    }
  }
};


}  // namespace detail

}  // namespace mg

}  // namespace cugraph

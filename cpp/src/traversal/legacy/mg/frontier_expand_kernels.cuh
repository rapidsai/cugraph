/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "vertex_binning.cuh"
#include <cugraph/legacy/graph.hpp>

namespace cugraph {

namespace mg {

namespace detail {

template <typename vertex_t, typename edge_t>
__device__ void write_to_frontier(vertex_t const* thread_frontier,
                                  int thread_frontier_count,
                                  vertex_t* block_frontier,
                                  int* block_frontier_count,
                                  vertex_t* output_frontier,
                                  edge_t* block_write_offset,
                                  edge_t* output_frontier_count)
{
  // Set frontier count for block to 0
  if (threadIdx.x == 0) { *block_frontier_count = 0; }
  __syncthreads();

  // Find out where to write the thread frontier to shared memory
  int thread_write_offset = atomicAdd(block_frontier_count, thread_frontier_count);
  for (int i = 0; i < thread_frontier_count; ++i) {
    block_frontier[i + thread_write_offset] = thread_frontier[i];
  }
  __syncthreads();

  // If the total number of frontiers for this block is 0 then return
  if (*block_frontier_count == 0) { return; }

  // Find out where to write the block frontier to global memory
  if (threadIdx.x == 0) {
    *block_write_offset = cugraph::detail::traversal::atomicAdd(
      output_frontier_count, static_cast<edge_t>(*block_frontier_count));
  }
  __syncthreads();

  // Write block frontier to global memory
  for (int i = threadIdx.x; i < (*block_frontier_count); i += blockDim.x) {
    output_frontier[(*block_write_offset) + i] = block_frontier[i];
  }
}

template <int BlockSize,
          int EdgesPerThread,
          typename vertex_t,
          typename edge_t,
          typename operator_t>
__global__ void block_per_vertex(edge_t const* offsets,
                                 vertex_t const* indices,
                                 vertex_t const* input_frontier,
                                 vertex_t input_frontier_count,
                                 vertex_t vertex_begin,
                                 vertex_t* output_frontier,
                                 edge_t* output_frontier_count,
                                 operator_t op)
{
  if (blockIdx.x >= input_frontier_count) { return; }

  __shared__ edge_t block_write_offset;
  __shared__ vertex_t block_frontier[BlockSize * EdgesPerThread];
  __shared__ int block_frontier_count;
  vertex_t thread_frontier[EdgesPerThread];

  vertex_t source        = input_frontier[blockIdx.x];
  edge_t beg_edge_offset = offsets[source];
  edge_t end_edge_offset = offsets[source + 1];

  edge_t edge_offset = threadIdx.x + beg_edge_offset;
  int num_iter       = (end_edge_offset - beg_edge_offset + BlockSize - 1) / BlockSize;

  int thread_frontier_count = 0;
  for (int i = 0; i < num_iter; ++i) {
    if (edge_offset < end_edge_offset) {
      vertex_t destination = indices[edge_offset];
      // If operator returns true then add to local frontier
      if (op(source + vertex_begin, destination)) {
        thread_frontier[thread_frontier_count++] = destination;
      }
    }
    bool is_last_iter = (i == (num_iter - 1));
    bool is_nth_iter  = (i % EdgesPerThread == 0);
    // Write to frontier every EdgesPerThread iterations
    // Or if it is the last iteration of the for loop
    if (is_nth_iter || is_last_iter) {
      write_to_frontier(thread_frontier,
                        thread_frontier_count,
                        block_frontier,
                        &block_frontier_count,
                        output_frontier,
                        &block_write_offset,
                        output_frontier_count);
      thread_frontier_count = 0;
    }
    edge_offset += blockDim.x;
  }
}

template <int BlockSize,
          int EdgesPerThread,
          typename vertex_t,
          typename edge_t,
          typename operator_t>
__global__ void kernel_per_vertex(edge_t const* offsets,
                                  vertex_t const* indices,
                                  vertex_t const* input_frontier,
                                  vertex_t input_frontier_count,
                                  vertex_t vertex_begin,
                                  vertex_t* output_frontier,
                                  edge_t* output_frontier_count,
                                  operator_t op)
{
  vertex_t current_vertex_index = 0;
  __shared__ edge_t block_write_offset;
  __shared__ vertex_t block_frontier[BlockSize * EdgesPerThread];
  __shared__ int block_frontier_count;

  edge_t stride = blockDim.x * gridDim.x;
  vertex_t thread_frontier[EdgesPerThread];

  while (current_vertex_index < input_frontier_count) {
    vertex_t source           = input_frontier[current_vertex_index];
    edge_t beg_block_offset   = offsets[source] + (blockIdx.x * blockDim.x);
    edge_t end_block_offset   = offsets[source + 1];
    int i                     = 0;
    int thread_frontier_count = 0;
    for (edge_t block_offset = beg_block_offset; block_offset < end_block_offset;
         block_offset += stride) {
      if (block_offset + threadIdx.x < end_block_offset) {
        vertex_t destination = indices[block_offset + threadIdx.x];
        if (op(source + vertex_begin, destination)) {
          thread_frontier[thread_frontier_count++] = destination;
        }
      }
      bool is_last_iter = (block_offset + blockDim.x >= end_block_offset);
      bool is_nth_iter  = (i % EdgesPerThread == 0);
      if (is_nth_iter || is_last_iter) {
        write_to_frontier(thread_frontier,
                          thread_frontier_count,
                          block_frontier,
                          &block_frontier_count,
                          output_frontier,
                          &block_write_offset,
                          output_frontier_count);
        thread_frontier_count = 0;
      }
      ++i;
    }
    ++current_vertex_index;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename operator_t>
void large_vertex_lb(cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                     DegreeBucket<vertex_t, edge_t>& bucket,
                     operator_t op,
                     vertex_t vertex_begin,
                     vertex_t* output_vertex_ids,
                     edge_t* output_vertex_ids_offset,
                     cudaStream_t stream)
{
  if (bucket.numberOfVertices != 0) {
    const int block_size = 1024;
    int block_count      = (1 << (bucket.ceilLogDegreeStart - 8));
    kernel_per_vertex<block_size, 2>
      <<<block_count, block_size, 0, stream>>>(graph.offsets,
                                               graph.indices,
                                               bucket.vertexIds,
                                               bucket.numberOfVertices,
                                               vertex_begin,
                                               output_vertex_ids,
                                               output_vertex_ids_offset,
                                               op);
    RAFT_CHECK_CUDA(stream);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename operator_t>
void medium_vertex_lb(cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                      DegreeBucket<vertex_t, edge_t>& bucket,
                      operator_t op,
                      vertex_t vertex_begin,
                      vertex_t* output_vertex_ids,
                      edge_t* output_vertex_ids_offset,
                      cudaStream_t stream)
{
  // Vertices with degrees 2^12 <= d < 2^16 are handled by this kernel
  // Block size of 1024 is chosen to reduce wasted threads for a vertex
  const int block_size = 1024;
  int block_count      = bucket.numberOfVertices;
  if (block_count != 0) {
    block_per_vertex<block_size, 2>
      <<<block_count, block_size, 0, stream>>>(graph.offsets,
                                               graph.indices,
                                               bucket.vertexIds,
                                               bucket.numberOfVertices,
                                               vertex_begin,
                                               output_vertex_ids,
                                               output_vertex_ids_offset,
                                               op);
    RAFT_CHECK_CUDA(stream);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename operator_t>
void small_vertex_lb(cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                     DegreeBucket<vertex_t, edge_t>& bucket,
                     operator_t op,
                     vertex_t vertex_begin,
                     vertex_t* output_vertex_ids,
                     edge_t* output_vertex_ids_offset,
                     cudaStream_t stream)
{
  int block_count = bucket.numberOfVertices;
  if (block_count == 0) { return; }
  // For vertices with degree <= 32 block size of 32 is chosen
  // For all vertices with degree d such that 2^x <= d < 2^x+1
  // the block size is chosen to be 2^x. This is done so that
  // vertices with degrees 1.5*2^x are also handled in a load
  // balanced way
  int block_size = 512;
  if (bucket.ceilLogDegreeEnd < 6) {
    block_size = 32;
    block_per_vertex<32, 8><<<block_count, block_size, 0, stream>>>(graph.offsets,
                                                                    graph.indices,
                                                                    bucket.vertexIds,
                                                                    bucket.numberOfVertices,
                                                                    vertex_begin,
                                                                    output_vertex_ids,
                                                                    output_vertex_ids_offset,
                                                                    op);
  } else if (bucket.ceilLogDegreeEnd < 8) {
    block_size = 64;
    block_per_vertex<64, 8><<<block_count, block_size, 0, stream>>>(graph.offsets,
                                                                    graph.indices,
                                                                    bucket.vertexIds,
                                                                    bucket.numberOfVertices,
                                                                    vertex_begin,
                                                                    output_vertex_ids,
                                                                    output_vertex_ids_offset,
                                                                    op);
  } else if (bucket.ceilLogDegreeEnd < 10) {
    block_size = 128;
    block_per_vertex<128, 8><<<block_count, block_size, 0, stream>>>(graph.offsets,
                                                                     graph.indices,
                                                                     bucket.vertexIds,
                                                                     bucket.numberOfVertices,
                                                                     vertex_begin,
                                                                     output_vertex_ids,
                                                                     output_vertex_ids_offset,
                                                                     op);
  } else if (bucket.ceilLogDegreeEnd < 12) {
    block_size = 512;
    block_per_vertex<512, 4><<<block_count, block_size, 0, stream>>>(graph.offsets,
                                                                     graph.indices,
                                                                     bucket.vertexIds,
                                                                     bucket.numberOfVertices,
                                                                     vertex_begin,
                                                                     output_vertex_ids,
                                                                     output_vertex_ids_offset,
                                                                     op);
  } else {
    block_size = 512;
    block_per_vertex<512, 4><<<block_count, block_size, 0, stream>>>(graph.offsets,
                                                                     graph.indices,
                                                                     bucket.vertexIds,
                                                                     bucket.numberOfVertices,
                                                                     vertex_begin,
                                                                     output_vertex_ids,
                                                                     output_vertex_ids_offset,
                                                                     op);
  }
  RAFT_CHECK_CUDA(stream);
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph

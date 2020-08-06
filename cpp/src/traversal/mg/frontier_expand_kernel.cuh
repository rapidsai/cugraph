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

#include "../traversal_common.cuh"

namespace cugraph {

namespace mg {

namespace detail {

template <int EdgesPerBlock, int Threads, int EdgesPerThread,
         typename vertex_t, typename edge_t, typename Operator>
__global__
void frontier_expand_kernel(
    edge_t * offsets,
    vertex_t * indices,
    vertex_t * input_frontier,
    vertex_t input_frontier_len,
    vertex_t vertex_begin,
    edge_t total_frontier_edge_count,
    edge_t * frontier_vertex_offset,
    edge_t * frontier_vertex_bucket_offset,
    vertex_t * output_frontier,
    edge_t * output_frontier_size,
    edge_t max_output_frontier_size,
    Operator op) {

  __shared__ union TempStorage
  {
    edge_t local_offsets[EdgesPerBlock];
    vertex_t frontier_vertices[EdgesPerBlock];
  } temp_storage;
  __shared__ int local_frontier_count;
  __shared__ edge_t block_write_offset;
  if (threadIdx.x == 0) {
    local_frontier_count = 0;
  }

  edge_t offset_beg = frontier_vertex_bucket_offset[blockIdx.x];
  edge_t offset_end = frontier_vertex_bucket_offset[blockIdx.x + 1];
  for (int i = threadIdx.x; i < (offset_end - offset_beg); i += blockDim.x) {
    temp_storage.local_offsets[i] = frontier_vertex_offset[i + offset_beg];
  }
  __syncthreads();

  vertex_t source[EdgesPerThread];
  vertex_t destination[EdgesPerThread];

  edge_t end_edge_index =
    min(total_frontier_edge_count,
        static_cast<edge_t>(EdgesPerBlock*(blockIdx.x+1)));
  int total_edges_processed = 0;
  for (edge_t edge_index = threadIdx.x + (EdgesPerBlock*blockIdx.x);
      edge_index < end_edge_index;
      edge_index += blockDim.x)
  {
    edge_t source_index =
      cugraph::detail::traversal::binsearch_maxle(
          temp_storage.local_offsets,
          edge_index,
          static_cast<edge_t>(0),
          (offset_end - offset_beg));
    source[total_edges_processed] = input_frontier[offset_beg + source_index];
    edge_t sub_edge_index = edge_index - temp_storage.local_offsets[source_index];
    edge_t vertex_offset = offsets[source[total_edges_processed] - vertex_begin];
    destination[total_edges_processed] = indices[vertex_offset + sub_edge_index];
    ++total_edges_processed;
  }
  //Make sure all usage of temp_storage.local_offsets is finished
  __syncthreads();

  //Write all acceptable frontier vertices to shared memory
  //local_frontier_count shared variable should contain the
  //total number of acceptable vertices
  for (edge_t i = 0; i < total_edges_processed; ++i) {
    if (op(source[i], destination[i])) {
      //local_output_frontier[local_frontier_count++] = destination[i];
      int write_index = atomicAdd(&local_frontier_count, 1);
      temp_storage.frontier_vertices[write_index] = destination[i];
    }
  }
  __syncthreads();

  //Find out from which point in global memory should the
  //block write frontier vertices to
  if (threadIdx.x == 0) {
    block_write_offset = cugraph::detail::traversal::atomicAdd(
        output_frontier_size,
        static_cast<edge_t>(local_frontier_count));
  }
  __syncthreads();

  //If the kernel tries to write beyond the capacity of
  //the output frontier then something has gone wrong
  //in the frontier creation operator supplied by user
  if (block_write_offset + local_frontier_count >=
      max_output_frontier_size) {
    printf("err : attempting to write out of bounds\n");
    return;
    //TODO : assert and end kernel?
  }

  //Write frontier to global memory
  for (int i = threadIdx.x;
      i < local_frontier_count;
      i += blockDim.x) {
    output_frontier[block_write_offset + i] =
      temp_storage.frontier_vertices[i];
  }

}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph

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

#include <graph.hpp>
#include "vertex_binning.cuh"

namespace cugraph {

namespace mg {

namespace detail {

template <typename VT, typename ET, typename WT, typename Operator>
__global__ void kernel_per_vertex_worker_weightless(cugraph::GraphCSRView<VT, ET, WT> graph,
                                                    VT *vertex_ids,
                                                    VT number_of_vertices,
                                                    Operator op)
{
  VT current_vertex_index = 0;
  ET tid                  = threadIdx.x + (blockIdx.x * blockDim.x);
  ET stride               = blockDim.x * gridDim.x;

  while (current_vertex_index < number_of_vertices) {
    VT source       = vertex_ids[current_vertex_index];
    ET offset_begin = graph.offsets[source];
    ET offset_end   = graph.offsets[source + 1];
    for (ET edge_index = tid + offset_begin; edge_index < offset_end; edge_index += stride) {
      op(source, graph.indices[edge_index]);
    }
    current_vertex_index++;
  }
}

template <typename VT, typename ET, typename WT, typename Operator>
void large_vertex_worker(cugraph::GraphCSRView<VT, ET, WT> const &graph,
                         DegreeBucket<VT, ET> &bucket,
                         Operator op,
                         cudaStream_t stream)
{
  int block_size  = 32;
  int block_count = bucket.numberOfVertices * (1 << (bucket.ceilLogDegreeStart - 5));
  if (bucket.numberOfVertices != 0) {
    kernel_per_vertex_worker_weightless<<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices, op);
  }
}

template <typename VT, typename ET, typename WT, typename Operator>
__global__ void block_per_vertex_worker_weightless(cugraph::GraphCSRView<VT, ET, WT> graph,
                                                   VT *vertex_ids,
                                                   VT number_of_vertices,
                                                   Operator op)
{
  VT current_vertex_index = blockIdx.x;
  if (current_vertex_index >= number_of_vertices) { return; }

  VT source = vertex_ids[current_vertex_index];
  for (ET edge_index = threadIdx.x + graph.offsets[source]; edge_index < graph.offsets[source + 1];
       edge_index += blockDim.x) {
    op(source, graph.indices[edge_index]);
  }
}

template <typename VT, typename ET, typename WT, typename Operator>
void medium_vertex_worker(cugraph::GraphCSRView<VT, ET, WT> const &graph,
                          DegreeBucket<VT, ET> &bucket,
                          Operator op,
                          cudaStream_t stream)
{
  // Vertices with degrees 2^12 <= d < 2^16 are handled by this kernel
  // Block size of 1024 is chosen to reduce wasted threads for a vertex
  int block_size  = 1024;
  int block_count = bucket.numberOfVertices;
  if (block_count != 0) {
    block_per_vertex_worker_weightless<<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices, op);
  }
}

template <typename VT, typename ET, typename WT, typename Operator>
void small_vertex_worker(cugraph::GraphCSRView<VT, ET, WT> const &graph,
                         DegreeBucket<VT, ET> &bucket,
                         Operator op,
                         cudaStream_t stream)
{
  int block_size = 512;
  if (bucket.ceilLogDegreeEnd < 6) {
    block_size = 32;
  } else if (bucket.ceilLogDegreeEnd < 8) {
    block_size = 64;
  } else if (bucket.ceilLogDegreeEnd < 10) {
    block_size = 128;
  } else if (bucket.ceilLogDegreeEnd < 12) {
    block_size = 512;
  }
  // For vertices with degree <= 32 block size of 32 is chosen
  // For all vertices with degree d such that 2^x <= d < 2^x+1
  // the block size is chosen to be 2^x. This is done so that
  // vertices with degrees 1.5*2^x are also handled in a load
  // balanced way
  int block_count = bucket.numberOfVertices;
  if (block_count != 0) {
    block_per_vertex_worker_weightless<<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices, op);
  }
}
////////////////////////////////////////////////////////////////////////////////
// Queue enabled kernels
////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE, typename VT, typename ET, typename WT, typename Operator>
__global__ void block_per_vertex_worker_weightless(cugraph::GraphCSRView<VT, ET, WT> graph,
                                                   VT *vertex_ids,
                                                   VT number_of_vertices,
                                                   VT *output_vertex_ids,
                                                   ET *output_vertex_ids_offset,
                                                   Operator op)
{
  //TODO : Increase REP
  const int REP = 4;
  VT local_frontier[REP];
  ET local_frontier_count = 0;
  ET local_frontier_offset = 0;
  ET local_frontier_offset_total = 0;
  VT current_vertex_index = blockIdx.x;
  __shared__ ET block_write_index;
  if (current_vertex_index >= number_of_vertices) { return; }

  VT source = vertex_ids[current_vertex_index];
  ET num_iter_this_vertex = (graph.offsets[source+1] - graph.offsets[source] + blockDim.x - 1)/blockDim.x;
  ET edge_index = threadIdx.x + graph.offsets[source];

  typedef cub::BlockScan<ET, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  bool flush = false;
  for (ET iter = 0; iter < num_iter_this_vertex; ++iter) {
    if (edge_index < graph.offsets[source+1]) {
      op(source, graph.indices[edge_index],
          local_frontier, &local_frontier_count);
    } else {
      local_frontier_count = 0;
    }
    if (iter % REP == 0) {
      local_frontier_offset_total = 0;
      BlockScan(temp_storage).ExclusiveSum(
          local_frontier_count,
          local_frontier_offset,
          local_frontier_offset_total);
      if (local_frontier_offset_total != 0) {
        if (threadIdx.x == 0) {
          block_write_index = cugraph::detail::traversal::atomicAdd(
              output_vertex_ids_offset, local_frontier_offset_total);
        }
        __syncthreads();
        for (ET i = 0; i < local_frontier_count; ++i) {
          output_vertex_ids[block_write_index + local_frontier_offset + i] = local_frontier[i];
        }
      }
      local_frontier_count = 0;
      local_frontier_offset = 0;
      local_frontier_offset_total = 0;
      flush = true;
    }
    edge_index += blockDim.x;
  }
  if (flush == false) {
    BlockScan(temp_storage).ExclusiveSum(
        local_frontier_count,
        local_frontier_offset,
        local_frontier_offset_total);
    if (local_frontier_offset_total != 0) {
      if (threadIdx.x == 0) {
        block_write_index = cugraph::detail::traversal::atomicAdd(
            output_vertex_ids_offset, local_frontier_offset_total);
      }
      __syncthreads();
      for (ET i = 0; i < local_frontier_count; ++i) {
        output_vertex_ids[block_write_index + local_frontier_offset + i] = local_frontier[i];
      }
    }
    local_frontier_count = 0;
    local_frontier_offset = 0;
    local_frontier_offset_total = 0;
  }
}


template <typename VT, typename ET, typename WT, typename Operator>
void small_vertex_worker(cugraph::GraphCSRView<VT, ET, WT> const &graph,
                         DegreeBucket<VT, ET> &bucket,
                         Operator op,
                         VT *output_vertex_ids,
                         ET *output_vertex_ids_offset,
                         cudaStream_t stream)
{
  int block_count = bucket.numberOfVertices;
  if (block_count == 0) {
    return;
  }
  // For vertices with degree <= 32 block size of 32 is chosen
  // For all vertices with degree d such that 2^x <= d < 2^x+1
  // the block size is chosen to be 2^x. This is done so that
  // vertices with degrees 1.5*2^x are also handled in a load
  // balanced way
  int block_size = 512;
  if (bucket.ceilLogDegreeEnd < 6) {
    block_size = 32;
    block_per_vertex_worker_weightless<32><<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices,
      output_vertex_ids, output_vertex_ids_offset, op);
  } else if (bucket.ceilLogDegreeEnd < 8) {
    block_size = 64;
    block_per_vertex_worker_weightless<64><<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices,
      output_vertex_ids, output_vertex_ids_offset, op);
  } else if (bucket.ceilLogDegreeEnd < 10) {
    block_size = 128;
    block_per_vertex_worker_weightless<128><<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices,
      output_vertex_ids, output_vertex_ids_offset, op);
  } else if (bucket.ceilLogDegreeEnd < 12) {
    block_size = 512;
    block_per_vertex_worker_weightless<512><<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices,
      output_vertex_ids, output_vertex_ids_offset, op);
  } else {
    block_size = 512;
    block_per_vertex_worker_weightless<512><<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices,
      output_vertex_ids, output_vertex_ids_offset, op);
  }
  CHECK_CUDA(stream);
}

template <typename VT, typename ET, typename WT, typename Operator>
void medium_vertex_worker(cugraph::GraphCSRView<VT, ET, WT> const &graph,
                          DegreeBucket<VT, ET> &bucket,
                          Operator op,
                          VT *output_vertex_ids,
                          ET *output_vertex_ids_offset,
                          cudaStream_t stream)
{
  // Vertices with degrees 2^12 <= d < 2^16 are handled by this kernel
  // Block size of 1024 is chosen to reduce wasted threads for a vertex
  const int block_size  = 1024;
  int block_count = bucket.numberOfVertices;
  if (block_count != 0) {
    block_per_vertex_worker_weightless<block_size><<<block_count, block_size, 0, stream>>>(
      graph, bucket.vertexIds, bucket.numberOfVertices,
      output_vertex_ids, output_vertex_ids_offset, op);
    CHECK_CUDA(stream);
  }
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph

/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
/** ---------------------------------------------------------------------------*
 * @brief Functions for computing the two hop neighbor pairs of a graph
 *
 * @file two_hop_neighbors.cuh
 * ---------------------------------------------------------------------------**/

#include <thrust/tuple.h>

#define MAXBLOCKS          65535
#define TWO_HOP_BLOCK_SIZE 512

template <typename edge_t>
struct degree_iterator {
  edge_t const* offsets;
  degree_iterator(edge_t const* _offsets) : offsets(_offsets) {}

  __host__ __device__ edge_t operator[](edge_t place)
  {
    return offsets[place + 1] - offsets[place];
  }
};

template <typename It, typename edge_t>
struct deref_functor {
  It iterator;
  deref_functor(It it) : iterator(it) {}

  __host__ __device__ edge_t operator()(edge_t in) { return iterator[in]; }
};

template <typename vertex_t>
struct self_loop_flagger {
  __host__ __device__ bool operator()(const thrust::tuple<vertex_t, vertex_t> pair)
  {
    if (thrust::get<0>(pair) == thrust::get<1>(pair)) return false;
    return true;
  }
};

template <typename edge_t>
__device__ edge_t binsearch_maxle(const edge_t* vec, const edge_t val, edge_t low, edge_t high)
{
  while (true) {
    if (low == high) return low;  // we know it exists
    if ((low + 1) == high) return (vec[high] <= val) ? high : low;

    edge_t mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

template <typename edge_t>
__global__ void compute_bucket_offsets_kernel(const edge_t* frontier_degrees_exclusive_sum,
                                              edge_t* bucket_offsets,
                                              const edge_t frontier_size,
                                              edge_t total_degree)
{
  edge_t end = ((total_degree - 1 + TWO_HOP_BLOCK_SIZE) / TWO_HOP_BLOCK_SIZE);

  for (edge_t bid = blockIdx.x * blockDim.x + threadIdx.x; bid <= end;
       bid += gridDim.x * blockDim.x) {
    edge_t eid = min(bid * TWO_HOP_BLOCK_SIZE, total_degree - 1);

    bucket_offsets[bid] =
      binsearch_maxle(frontier_degrees_exclusive_sum, eid, edge_t{0}, frontier_size - 1);
  }
}

template <typename vertex_t, typename edge_t>
__global__ void scatter_expand_kernel(const edge_t* exsum_degree,
                                      const vertex_t* indices,
                                      const edge_t* offsets,
                                      const edge_t* bucket_offsets,
                                      vertex_t num_verts,
                                      edge_t max_item,
                                      edge_t max_block,
                                      vertex_t* output_first,
                                      vertex_t* output_second)
{
  __shared__ edge_t blockRange[2];
  for (edge_t bid = blockIdx.x; bid < max_block; bid += gridDim.x) {
    // Copy the start and end of the buckets range into shared memory
    if (threadIdx.x == 0) {
      blockRange[0] = bucket_offsets[bid];
      blockRange[1] = bucket_offsets[bid + 1];
    }
    __syncthreads();

    // Get the global thread id (for this virtual block)
    edge_t tid = bid * blockDim.x + threadIdx.x;
    if (tid < max_item) {
      edge_t sourceIdx    = binsearch_maxle(exsum_degree, tid, blockRange[0], blockRange[1]);
      vertex_t sourceId   = indices[sourceIdx];
      edge_t itemRank     = tid - exsum_degree[sourceIdx];
      output_second[tid]  = indices[offsets[sourceId] + itemRank];
      edge_t baseSourceId = binsearch_maxle(offsets, sourceIdx, edge_t{0}, edge_t{num_verts});
      output_first[tid]   = baseSourceId;
    }
  }
}

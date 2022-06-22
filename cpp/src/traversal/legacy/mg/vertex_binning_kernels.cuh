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

#include "../traversal_common.cuh"
#include <rmm/device_vector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace cugraph {

namespace mg {

namespace detail {

template <typename degree_t>
__device__ inline typename std::enable_if<(sizeof(degree_t) == 4), int>::type ceilLog2_p1(
  degree_t val)
{
  return BitsPWrd<degree_t> - __clz(val) + (__popc(val) > 1);
}

template <typename degree_t>
__device__ inline typename std::enable_if<(sizeof(degree_t) == 8), int>::type ceilLog2_p1(
  degree_t val)
{
  return BitsPWrd<degree_t> - __clzll(val) + (__popcll(val) > 1);
}

template <typename return_t>
__global__ void simple_fill(return_t* bin0, return_t* bin1, return_t count)
{
  for (return_t i = 0; i < count; i++) {
    bin0[i] = 0;
    bin1[i] = 0;
  }
}

template <typename return_t>
__global__ void exclusive_scan(return_t* data, return_t* out)
{
  constexpr int BinCount = NumberBins<return_t>;
  return_t lData[BinCount];
  thrust::exclusive_scan(thrust::seq, data, data + BinCount, lData);
  for (int i = 0; i < BinCount; ++i) {
    out[i]  = lData[i];
    data[i] = lData[i];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Queue enabled kernels
////////////////////////////////////////////////////////////////////////////////

// Given the CSR offsets of vertices and the related active bit map
// count the number of vertices that belong to a particular bin where
// vertex with degree d such that 2^x < d <= 2^x+1 belong to bin (x+1)
// Vertices with degree 0 are counted in bin 0
// In this function, any id in vertex_ids array is only acceptable as long
// as its value is between vertex_begin and vertex_end
template <typename vertex_t, typename edge_t>
__global__ void count_bin_sizes(edge_t* bins,
                                edge_t const* offsets,
                                vertex_t const* vertex_ids,
                                edge_t const vertex_id_count,
                                vertex_t vertex_begin,
                                vertex_t vertex_end)
{
  using cugraph::detail::traversal::atomicAdd;
  constexpr int BinCount = NumberBins<edge_t>;
  __shared__ edge_t lBin[BinCount];
  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    lBin[i] = 0;
  }
  __syncthreads();

  for (vertex_t i = threadIdx.x + (blockIdx.x * blockDim.x); i < vertex_id_count;
       i += gridDim.x * blockDim.x) {
    auto source = vertex_ids[i];
    if ((source >= vertex_begin) && (source < vertex_end)) {
      // Take care of OPG partitioning
      // source logical vertex resides from offsets[source - vertex_begin]
      // to offsets[source - vertex_begin + 1]
      source -= vertex_begin;
      auto degree = offsets[source + 1] - offsets[source];
      atomicAdd(lBin + ceilLog2_p1(degree), edge_t{1});
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    atomicAdd(bins + i, lBin[i]);
  }
}

// Bin vertices to the appropriate bins by taking into account
// the starting offsets calculated by count_bin_sizes
template <typename vertex_t, typename edge_t>
__global__ void create_vertex_bins(vertex_t* out_vertex_ids,
                                   edge_t* bin_offsets,
                                   edge_t const* offsets,
                                   vertex_t* in_vertex_ids,
                                   edge_t const vertex_id_count,
                                   vertex_t vertex_begin,
                                   vertex_t vertex_end)
{
  using cugraph::detail::traversal::atomicAdd;
  constexpr int BinCount = NumberBins<edge_t>;
  __shared__ edge_t lBin[BinCount];
  __shared__ int lPos[BinCount];
  if (threadIdx.x < BinCount) {
    lBin[threadIdx.x] = 0;
    lPos[threadIdx.x] = 0;
  }
  __syncthreads();

  vertex_t vertex_index = (threadIdx.x + blockIdx.x * blockDim.x);
  bool is_valid_vertex  = (vertex_index < vertex_id_count);
  vertex_t source;

  if (is_valid_vertex) {
    source          = in_vertex_ids[vertex_index];
    is_valid_vertex = ((source >= vertex_begin) && (source < vertex_end));
    source -= vertex_begin;
  }

  int threadBin;
  edge_t threadPos;
  if (is_valid_vertex) {
    threadBin = ceilLog2_p1(offsets[source + 1] - offsets[source]);
    threadPos = atomicAdd(lBin + threadBin, edge_t{1});
  }
  __syncthreads();

  if (threadIdx.x < BinCount) {
    lPos[threadIdx.x] = atomicAdd(bin_offsets + threadIdx.x, lBin[threadIdx.x]);
  }
  __syncthreads();

  if (is_valid_vertex) { out_vertex_ids[lPos[threadBin] + threadPos] = source; }
}

template <typename vertex_t, typename edge_t>
void bin_vertices(rmm::device_vector<vertex_t>& input_vertex_ids,
                  vertex_t input_vertex_ids_len,
                  rmm::device_vector<vertex_t>& reorganized_vertex_ids,
                  rmm::device_vector<edge_t>& bin_count_offsets,
                  rmm::device_vector<edge_t>& bin_count,
                  edge_t* offsets,
                  vertex_t vertex_begin,
                  vertex_t vertex_end,
                  cudaStream_t stream)
{
  simple_fill<edge_t><<<1, 1, 0, stream>>>(
    bin_count_offsets.data().get(), bin_count.data().get(), static_cast<edge_t>(bin_count.size()));

  const uint32_t BLOCK_SIZE = 512;
  uint32_t blocks           = ((input_vertex_ids_len) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  count_bin_sizes<edge_t>
    <<<blocks, BLOCK_SIZE, 0, stream>>>(bin_count.data().get(),
                                        offsets,
                                        input_vertex_ids.data().get(),
                                        static_cast<edge_t>(input_vertex_ids_len),
                                        vertex_begin,
                                        vertex_end);

  exclusive_scan<<<1, 1, 0, stream>>>(bin_count.data().get(), bin_count_offsets.data().get());

  create_vertex_bins<vertex_t, edge_t>
    <<<blocks, BLOCK_SIZE, 0, stream>>>(reorganized_vertex_ids.data().get(),
                                        bin_count.data().get(),
                                        offsets,
                                        input_vertex_ids.data().get(),
                                        static_cast<edge_t>(input_vertex_ids_len),
                                        vertex_begin,
                                        vertex_end);
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph

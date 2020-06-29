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

#ifndef BIN_KERNELS_CUH
#define BIN_KERNELS_CUH

#include <rmm/thrust_rmm_allocator.h>

namespace cugraph {

namespace detail {

namespace opg {

template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t)*8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t)*8 + 1;

template <typename degree_t>
__device__ inline
typename std::enable_if<(sizeof(degree_t) == 4), int>::type
ceilLog2_p1(degree_t val) {
  return BitsPWrd<degree_t> - __clz(val) + (__popc(val) > 1);
}

template <typename degree_t>
__device__ inline
typename std::enable_if<(sizeof(degree_t) == 8), int>::type
ceilLog2_p1(degree_t val) {
  return BitsPWrd<degree_t> - __clzll(val) + (__popcll(val) > 1);
}

template <typename T>
__global__ void
exclusive_scan(T* data, T* out) {
  constexpr int BinCount = NumberBins<T>;
  T lData[BinCount];
  thrust::exclusive_scan(thrust::seq, data, data + BinCount, lData);
  for (int i = 0; i < BinCount; ++i) {
    out[i] = lData[i];
    data[i] = lData[i];
  }
}

//Return true if the nth bit of an array is set to 1
template <typename T>
__device__
bool is_nth_bit_set(
    unsigned * bitmap,
    T index) {
  return bitmap[index/BitsPWrd<unsigned>] & (unsigned{1} << (index % BitsPWrd<unsigned>));
}

//Given the CSR offsets of vertices and the related active bit map
//count the number of vertices that belong to a particular bin where
//vertex with degree d such that 2^x < d <= 2^x+1 belong to bin (x+1)
//Vertices with degree 0 are counted in bin 0
template <typename VT, typename ET>
__global__
void count_bin_sizes(
    ET *bins,
    unsigned *active_bitmap,
    ET const *offsets,
    VT vertex_begin,
    VT vertex_end) {
  constexpr int BinCount = NumberBins<ET>;
  __shared__ ET lBin[BinCount];
  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    lBin[i] = 0;
  }
  __syncthreads();

  for (VT i =  threadIdx.x + (blockIdx.x*blockDim.x);
      i < (vertex_end - vertex_begin); i += gridDim.x*blockDim.x) {
    if (is_nth_bit_set(active_bitmap, vertex_begin + i)) {
      atomicAdd(lBin + ceilLog2_p1(offsets[i+1] - offsets[i]), 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    atomicAdd(bins + i, lBin[i]);
  }
}

//Given the CSR offsets of vertices count the number of vertices that
//belong to a particular bin where vertex with degree d such that
//2^x < d <= 2^x+1 belong to bin (x+1). Vertices with degree 0 are counted in
//bin 0
template <typename VT, typename ET>
__global__
void count_bin_sizes(
    ET *bins,
    ET const *offsets,
    VT vertex_begin,
    VT vertex_end) {
  constexpr int BinCount = NumberBins<ET>;
  __shared__ ET lBin[BinCount];
  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    lBin[i] = 0;
  }
  __syncthreads();

  for (VT i =  threadIdx.x + (blockIdx.x*blockDim.x);
      i < (vertex_end - vertex_begin); i += gridDim.x*blockDim.x) {
    atomicAdd(lBin + ceilLog2_p1(offsets[i+1] - offsets[i]), 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    atomicAdd(bins + i, lBin[i]);
  }
}

//Bin vertices to the appropriate bins by taking into account
//the starting offsets calculated by count_bin_sizes
template <typename VT, typename ET>
__global__
void create_vertex_bins(
    VT * reorg_vertices,
    ET * bin_offsets,
    unsigned *active_bitmap,
    ET const *offsets,
    VT vertex_begin,
    VT vertex_end) {
  constexpr int BinCount = NumberBins<ET>;
  __shared__ ET lBin[BinCount];
  __shared__ int lPos[BinCount];
  if (threadIdx.x < BinCount) {
    lBin[threadIdx.x] = 0; lPos[threadIdx.x] = 0;
  }
  __syncthreads();

  VT vertex_id = (threadIdx.x + blockIdx.x*blockDim.x);
  bool is_valid_vertex = (vertex_id < (vertex_end - vertex_begin)) &&
    (is_nth_bit_set(active_bitmap, vertex_begin + vertex_id));

  int threadBin;
  ET threadPos;
  if (is_valid_vertex) {
    threadBin = ceilLog2_p1(offsets[vertex_id+1] - offsets[vertex_id]);
    threadPos = atomicAdd(lBin + threadBin, 1);
  }
  __syncthreads();

  if (threadIdx.x < BinCount) {
    lPos[threadIdx.x] = atomicAdd(bin_offsets + threadIdx.x, lBin[threadIdx.x]);
  }
  __syncthreads();

  if (is_valid_vertex) {
    reorg_vertices[lPos[threadBin] + threadPos] = vertex_id;
  }
}

//Bin vertices to the appropriate bins by taking into account
//the starting offsets calculated by count_bin_sizes
template <typename VT, typename ET>
__global__
void create_vertex_bins(
    VT * reorg_vertices,
    ET * bin_offsets,
    ET const *offsets,
    VT vertex_begin,
    VT vertex_end) {
  constexpr int BinCount = NumberBins<ET>;
  __shared__ ET lBin[BinCount];
  __shared__ int lPos[BinCount];
  if (threadIdx.x < BinCount) {
    lBin[threadIdx.x] = 0; lPos[threadIdx.x] = 0;
  }
  __syncthreads();

  VT vertex_id = (threadIdx.x + blockIdx.x*blockDim.x);
  bool is_valid_vertex = (vertex_id < (vertex_end - vertex_begin));

  int threadBin;
  ET threadPos;
  if (is_valid_vertex) {
    threadBin = ceilLog2_p1(offsets[vertex_id+1] - offsets[vertex_id]);
    threadPos = atomicAdd(lBin + threadBin, 1);
  }
  __syncthreads();

  if (threadIdx.x < BinCount) {
    lPos[threadIdx.x] = atomicAdd(bin_offsets + threadIdx.x, lBin[threadIdx.x]);
  }
  __syncthreads();

  if (is_valid_vertex) {
    reorg_vertices[lPos[threadBin] + threadPos] = vertex_id;
  }
}

template <typename VT, typename ET>
void bin_vertices(
    rmm::device_vector<VT> &reorg_vertices,
    rmm::device_vector<ET> &bin_count_offsets,
    rmm::device_vector<ET> &bin_count,
    unsigned *active_bitmap,
    ET *offsets,
    VT vertex_begin,
    VT vertex_end,
    cudaStream_t stream) {

  const unsigned BLOCK_SIZE = 512;
  unsigned blocks = ((vertex_end - vertex_begin) + BLOCK_SIZE - 1)/BLOCK_SIZE;
  if (active_bitmap != nullptr) {
    count_bin_sizes<ET><<<blocks, BLOCK_SIZE, 0, stream>>>(
        bin_count.data().get(),
        active_bitmap,
        offsets,
        vertex_begin,
        vertex_end);
  } else {
    count_bin_sizes<ET><<<blocks, BLOCK_SIZE, 0, stream>>>(
        bin_count.data().get(),
        offsets,
        vertex_begin,
        vertex_end);
  }

  exclusive_scan<<<1,1,0,stream>>>(bin_count.data().get(), bin_count_offsets.data().get());

  VT vertex_count = bin_count[bin_count.size()-1];
  reorg_vertices.resize(vertex_count);

  if (active_bitmap != nullptr) {
    create_vertex_bins<VT, ET><<<blocks, BLOCK_SIZE, 0, stream>>>(
        reorg_vertices.data().get(),
        bin_count.data().get(),
        active_bitmap,
        offsets,
        vertex_begin,
        vertex_end);
  } else {
    create_vertex_bins<VT, ET><<<blocks, BLOCK_SIZE, 0, stream>>>(
        reorg_vertices.data().get(),
        bin_count.data().get(),
        offsets,
        vertex_begin,
        vertex_end);
  }

}

}//namespace opg

}//namespace detail

}//namespace cugraph

#endif //BIN_KERNELS_CUH

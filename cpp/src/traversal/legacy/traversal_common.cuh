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

#pragma once

#include <cub/cub.cuh>
#include <cugraph/utilities/error.hpp>

#include <thrust/scan.h>

#define MAXBLOCKS 65535
#define WARP_SIZE 32
#define INT_SIZE  32

//
// Bottom up macros
//

#define FILL_UNVISITED_QUEUE_DIMX 256

#define COUNT_UNVISITED_EDGES_DIMX 256

#define MAIN_BOTTOMUP_DIMX   256
#define MAIN_BOTTOMUP_NWARPS (MAIN_BOTTOMUP_DIMX / WARP_SIZE)

#define LARGE_BOTTOMUP_DIMX 256

// Number of edges processed in the main bottom up kernel
#define MAIN_BOTTOMUP_MAX_EDGES 6

// Power of 2 < 32 (strict <)
#define BOTTOM_UP_LOGICAL_WARP_SIZE 4

//
// Top down macros
//

// We will precompute the results the binsearch_maxle every
// TOP_DOWN_BUCKET_SIZE edges
#define TOP_DOWN_BUCKET_SIZE 32

// DimX of the kernel
#define TOP_DOWN_EXPAND_DIMX 256

// TOP_DOWN_EXPAND_DIMX edges -> NBUCKETS_PER_BLOCK buckets
#define NBUCKETS_PER_BLOCK (TOP_DOWN_EXPAND_DIMX / TOP_DOWN_BUCKET_SIZE)

// How many items_per_thread we can process with one bucket_offset loading
// the -1 is here because we need the +1 offset
#define MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD (TOP_DOWN_BUCKET_SIZE - 1)

// instruction parallelism
// for how many edges will we create instruction parallelism
#define TOP_DOWN_BATCH_SIZE 2

#define COMPUTE_BUCKET_OFFSETS_DIMX 512

// Other macros

#define FLAG_ISOLATED_VERTICES_DIMX 128

// Number of vertices handled by one thread
// Must be power of 2, lower than 32
#define FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD 4

// Number of threads involved in the "construction" of one int in the bitset
#define FLAG_ISOLATED_VERTICES_THREADS_PER_INT \
  (INT_SIZE / FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD)

//
// Parameters of the heuristic to switch between bottomup/topdown
// Finite machine described in
// http://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf
//

namespace cugraph {
namespace detail {
namespace traversal {

//
// gives the equivalent vectors from a type
// for the max val, would be better to use numeric_limits<>::max() once
// cpp11 is allowed in nvgraph
//

template <typename>
struct vec_t {
  typedef int4 vec4;
  typedef int2 vec2;
};

template <>
struct vec_t<int> {
  typedef int4 vec4;
  typedef int2 vec2;
  static const int max = std::numeric_limits<int>::max();
};

template <>
struct vec_t<long> {
  typedef long4 vec4;
  typedef long2 vec2;
  static const long max = std::numeric_limits<long>::max();
};

template <>
struct vec_t<unsigned> {
  typedef uint4 vec4;
  typedef uint2 vec2;
  static const unsigned max = std::numeric_limits<unsigned>::max();
};

template <>
struct vec_t<long long int> {
  typedef longlong4 vec4;
  typedef longlong2 vec2;
  static const long long int max = std::numeric_limits<long long int>::max();
};

template <>
struct vec_t<float> {
  typedef float4 vec4;
  typedef float2 vec2;
  static constexpr float max = std::numeric_limits<float>::max();
};

template <>
struct vec_t<double> {
  typedef double4 vec4;
  typedef double2 vec2;
  static constexpr double max = std::numeric_limits<double>::max();
};

//
// ------------------------- Helper device functions -------------------
//

__forceinline__ __device__ int getMaskNRightmostBitSet(int n)
{
  if (n == INT_SIZE) return (~0);
  int mask = (1 << n) - 1;
  return mask;
}

__forceinline__ __device__ int getMaskNLeftmostBitSet(int n)
{
  if (n == 0) return 0;
  int mask = ~((1 << (INT_SIZE - n)) - 1);
  return mask;
}

__forceinline__ __device__ int getNextZeroBit(int& val)
{
  int ibit = __ffs(~val) - 1;
  val |= (1 << ibit);

  return ibit;
}

struct BitwiseAnd {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return (a & b);
  }
};
struct BitwiseOr {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return (a | b);
  }
};

template <typename ValueType, typename SizeType>
__global__ void fill_vec_kernel(ValueType* vec, SizeType n, ValueType val)
{
  for (SizeType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    vec[idx] = val;
}

template <typename ValueType, typename SizeType>
void fill_vec(ValueType* vec, SizeType n, ValueType val, cudaStream_t stream)
{
  dim3 grid, block;
  block.x = 256;
  grid.x  = (n + block.x - 1) / block.x;

  fill_vec_kernel<<<grid, block, 0, stream>>>(vec, n, val);
  RAFT_CHECK_CUDA(stream);
}

template <typename IndexType>
__device__ IndexType
binsearch_maxle(const IndexType* vec, const IndexType val, IndexType low, IndexType high)
{
  while (true) {
    if (low == high) return low;  // we know it exists
    if ((low + 1) == high) return (vec[high] <= val) ? high : low;

    IndexType mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

// FIXME: The atomicAdd wrappers should be moved to RAFT

template <typename T>
__device__ static __forceinline__ T atomicAdd(T* addr, T val)
{
  return ::atomicAdd(addr, val);
}

template <>
__device__ __forceinline__ int64_t atomicAdd<int64_t>(int64_t* addr, int64_t val)
{
  static_assert(sizeof(int64_t) == sizeof(unsigned long long),
                "sizeof(int64_t) != sizeof(unsigned long long). Can't use atomicAdd");

  return ::atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(val));
}

__device__ static __forceinline__ float atomicMin(float* addr, float val)
{
  int* addr_as_int = (int*)addr;
  int old          = *addr_as_int;
  int expected;
  do {
    expected = old;
    old =
      ::atomicCAS(addr_as_int, expected, __float_as_int(::fminf(val, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

__device__ static __forceinline__ double atomicMin(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old     = ::atomicCAS(
      address_as_ull, assumed, __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

template <typename IndexType>
__global__ void flag_isolated_vertices_kernel(IndexType n,
                                              int* isolated_bmap,
                                              const IndexType* row_ptr,
                                              IndexType* degrees,
                                              IndexType* nisolated)
{
  typedef cub::BlockLoad<IndexType,
                         FLAG_ISOLATED_VERTICES_DIMX,
                         FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
    BlockLoad;
  typedef cub::BlockStore<IndexType,
                          FLAG_ISOLATED_VERTICES_DIMX,
                          FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD,
                          cub::BLOCK_STORE_WARP_TRANSPOSE>
    BlockStore;
  typedef cub::BlockReduce<IndexType, FLAG_ISOLATED_VERTICES_DIMX> BlockReduce;
  typedef cub::WarpReduce<int, FLAG_ISOLATED_VERTICES_THREADS_PER_INT> WarpReduce;

  __shared__ typename BlockLoad::TempStorage load_temp_storage;
  __shared__ typename BlockStore::TempStorage store_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_reduce_temp_storage;

  __shared__ typename WarpReduce::TempStorage
    warp_reduce_temp_storage[FLAG_ISOLATED_VERTICES_DIMX / FLAG_ISOLATED_VERTICES_THREADS_PER_INT];

  __shared__ IndexType row_ptr_tail[FLAG_ISOLATED_VERTICES_DIMX];

  for (IndexType block_off = FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD * (blockDim.x * blockIdx.x);
       block_off < n;
       block_off += FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD * (blockDim.x * gridDim.x)) {
    IndexType thread_off = block_off + FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD * threadIdx.x;
    IndexType last_node_thread = thread_off + FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1;

    IndexType thread_row_ptr[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD];
    IndexType block_valid_items = n - block_off + 1;  //+1, we need row_ptr[last_node+1]

    BlockLoad(load_temp_storage).Load(row_ptr + block_off, thread_row_ptr, block_valid_items, -1);

    // To compute 4 degrees, we need 5 values of row_ptr
    // Saving the "5th" value in shared memory for previous thread to use
    if (threadIdx.x > 0) { row_ptr_tail[threadIdx.x - 1] = thread_row_ptr[0]; }

    // If this is the last thread, it needs to load its row ptr tail value
    if (threadIdx.x == (FLAG_ISOLATED_VERTICES_DIMX - 1) && last_node_thread < n) {
      row_ptr_tail[threadIdx.x] = row_ptr[last_node_thread + 1];
    }
    __syncthreads();  // we may reuse temp_storage

    int local_isolated_bmap = 0;

    IndexType imax = (n > thread_off) ? (n - thread_off) : 0;

    IndexType local_degree[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD];

#pragma unroll
    for (int i = 0; i < (FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1); ++i) {
      IndexType degree = local_degree[i] = thread_row_ptr[i + 1] - thread_row_ptr[i];

      if (i < imax) local_isolated_bmap |= ((degree == 0) << i);
    }

    if (last_node_thread < n) {
      IndexType degree = local_degree[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1] =
        row_ptr_tail[threadIdx.x] - thread_row_ptr[FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1];

      local_isolated_bmap |= ((degree == 0) << (FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD - 1));
    }

    local_isolated_bmap <<= (thread_off % INT_SIZE);

    IndexType local_nisolated = __popc(local_isolated_bmap);

    // We need local_nisolated and local_isolated_bmap to be ready for next
    // steps
    __syncthreads();

    IndexType total_nisolated = BlockReduce(block_reduce_temp_storage).Sum(local_nisolated);

    if (threadIdx.x == 0 && total_nisolated) { traversal::atomicAdd(nisolated, total_nisolated); }

    int logicalwarpid = threadIdx.x / FLAG_ISOLATED_VERTICES_THREADS_PER_INT;

    // Building int for bmap
    int int_aggregate_isolated_bmap =
      WarpReduce(warp_reduce_temp_storage[logicalwarpid]).Reduce(local_isolated_bmap, BitwiseOr());

    int is_head_of_visited_int = ((threadIdx.x % (FLAG_ISOLATED_VERTICES_THREADS_PER_INT)) == 0);
    if (is_head_of_visited_int && (thread_off / INT_SIZE) < (n + INT_SIZE - 1) / INT_SIZE) {
      isolated_bmap[thread_off / INT_SIZE] = int_aggregate_isolated_bmap;
    }

    BlockStore(store_temp_storage).Store(degrees + block_off, local_degree, block_valid_items - 1);
  }
}

template <typename IndexType>
void flag_isolated_vertices(IndexType n,
                            int* isolated_bmap,
                            const IndexType* row_ptr,
                            IndexType* degrees,
                            IndexType* nisolated,
                            cudaStream_t m_stream)
{
  dim3 grid, block;
  block.x = FLAG_ISOLATED_VERTICES_DIMX;

  grid.x = min((IndexType)MAXBLOCKS,
               (n / FLAG_ISOLATED_VERTICES_VERTICES_PER_THREAD + 1 + block.x - 1) / block.x);

  flag_isolated_vertices_kernel<<<grid, block, 0, m_stream>>>(
    n, isolated_bmap, row_ptr, degrees, nisolated);
  RAFT_CHECK_CUDA(m_stream);
}

template <typename IndexType>
__global__ void set_frontier_degree_kernel(IndexType* frontier_degree,
                                           IndexType* frontier,
                                           const IndexType* degree,
                                           IndexType n)
{
  for (IndexType idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    IndexType u          = frontier[idx];
    frontier_degree[idx] = degree[u];
  }
}

template <typename IndexType>
void set_frontier_degree(IndexType* frontier_degree,
                         IndexType* frontier,
                         const IndexType* degree,
                         IndexType n,
                         cudaStream_t m_stream)
{
  dim3 grid, block;
  block.x = 256;
  grid.x  = min((n + block.x - 1) / block.x, (IndexType)MAXBLOCKS);
  set_frontier_degree_kernel<<<grid, block, 0, m_stream>>>(frontier_degree, frontier, degree, n);
  RAFT_CHECK_CUDA(m_stream);
}

template <typename IndexType>
void exclusive_sum(void* d_temp_storage,
                   size_t temp_storage_bytes,
                   IndexType* d_in,
                   IndexType* d_out,
                   IndexType num_items,
                   cudaStream_t m_stream)
{
  if (num_items <= 1) return;  // DeviceScan fails if n==1
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, m_stream);
}

template <typename IndexType>
void exclusive_sum(IndexType* d_in, IndexType* d_out, IndexType num_items, cudaStream_t m_stream)
{
  if (num_items <= 1) return;  // DeviceScan fails if n==1
  thrust::exclusive_scan(rmm::exec_policy(m_stream), d_in, d_in + num_items, d_out);
}

//
// compute_bucket_offsets_kernel
// simply compute the position in the frontier corresponding all valid edges
// with index=TOP_DOWN_BUCKET_SIZE * k, k integer
//

template <typename IndexType>
__global__ void compute_bucket_offsets_kernel(const IndexType* frontier_degrees_exclusive_sum,
                                              IndexType* bucket_offsets,
                                              const IndexType frontier_size,
                                              IndexType total_degree)
{
  IndexType end =
    ((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) / TOP_DOWN_EXPAND_DIMX * NBUCKETS_PER_BLOCK + 1);

  for (IndexType bid = blockIdx.x * blockDim.x + threadIdx.x; bid <= end;
       bid += gridDim.x * blockDim.x) {
    IndexType eid = min(bid * TOP_DOWN_BUCKET_SIZE, total_degree - 1);

    bucket_offsets[bid] =
      binsearch_maxle(frontier_degrees_exclusive_sum, eid, (IndexType)0, frontier_size - 1);
  }
}

template <typename IndexType>
void compute_bucket_offsets(IndexType* cumul,
                            IndexType* bucket_offsets,
                            IndexType frontier_size,
                            IndexType total_degree,
                            cudaStream_t m_stream)
{
  dim3 grid, block;
  block.x = COMPUTE_BUCKET_OFFSETS_DIMX;

  grid.x =
    min((IndexType)MAXBLOCKS,
        ((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) / TOP_DOWN_EXPAND_DIMX * NBUCKETS_PER_BLOCK + 1 +
         block.x - 1) /
          block.x);

  compute_bucket_offsets_kernel<<<grid, block, 0, m_stream>>>(
    cumul, bucket_offsets, frontier_size, total_degree);
  RAFT_CHECK_CUDA(m_stream);
}
}  // namespace traversal
}  // namespace detail
}  // namespace cugraph

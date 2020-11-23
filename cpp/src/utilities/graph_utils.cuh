/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Interanl helper functions
// Author: Alex Fender afender@nvidia.com
#pragma once

#include <utilities/error.hpp>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/device_atomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cugraph {
namespace detail {

//#define DEBUG 1
#define CUDA_MAX_BLOCKS 65535
#define CUDA_MAX_KERNEL_THREADS 256  // kernel will launch at most 256 threads per block
#define US

template <typename count_t, typename index_t, typename value_t>
__inline__ __device__ value_t parallel_prefix_sum(count_t n, index_t const *ind, value_t const *w)
{
  count_t i, j, mn;
  value_t v, last;
  value_t sum = 0.0;
  bool valid;

  // Parallel prefix sum (using __shfl)
  mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x);  // n in multiple of blockDim.x
  for (i = threadIdx.x; i < mn; i += blockDim.x) {
    // All threads (especially the last one) must always participate
    // in the shfl instruction, otherwise their sum will be undefined.
    // So, the loop stopping condition is based on multiple of n in loop increments,
    // so that all threads enter into the loop and inside we make sure we do not
    // read out of bounds memory checking for the actual size n.

    // check if the thread is valid
    valid = i < n;

    // Notice that the last thread is used to propagate the prefix sum.
    // For all the threads, in the first iteration the last is 0, in the following
    // iterations it is the value at the last thread of the previous iterations.

    // get the value of the last thread
    last = __shfl_sync(raft::warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

    // if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    // do prefix sum (of size warpSize=blockDim.x =< 32)
    for (j = 1; j < blockDim.x; j *= 2) {
      v = __shfl_up_sync(raft::warp_full_mask(), sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    // shift by last
    sum += last;
    // notice that no __threadfence or __syncthreads are needed in this implementation
  }
  // get the value of the last thread (to all threads)
  last = __shfl_sync(raft::warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

  return last;
}

// dot
template <typename T>
T dot(size_t n, T *x, T *y)
{
  cudaStream_t stream{nullptr};
  T result = thrust::inner_product(rmm::exec_policy(stream)->on(stream),
                                   thrust::device_pointer_cast(x),
                                   thrust::device_pointer_cast(x + n),
                                   thrust::device_pointer_cast(y),
                                   0.0f);
  CHECK_CUDA(stream);
  return result;
}

// axpy
template <typename T>
struct axpy_functor : public thrust::binary_function<T, T, T> {
  const T a;
  axpy_functor(T _a) : a(_a) {}
  __host__ __device__ T operator()(const T &x, const T &y) const { return a * x + y; }
};

template <typename T>
void axpy(size_t n, T a, T *x, T *y)
{
  cudaStream_t stream{nullptr};
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::device_pointer_cast(x),
                    thrust::device_pointer_cast(x + n),
                    thrust::device_pointer_cast(y),
                    thrust::device_pointer_cast(y),
                    axpy_functor<T>(a));
  CHECK_CUDA(stream);
}

// norm
template <typename T>
struct square {
  __host__ __device__ T operator()(const T &x) const { return x * x; }
};

template <typename T>
T nrm2(size_t n, T *x)
{
  cudaStream_t stream{nullptr};
  T init   = 0;
  T result = std::sqrt(thrust::transform_reduce(rmm::exec_policy(stream)->on(stream),
                                                thrust::device_pointer_cast(x),
                                                thrust::device_pointer_cast(x + n),
                                                square<T>(),
                                                init,
                                                thrust::plus<T>()));
  CHECK_CUDA(stream);
  return result;
}

template <typename T>
T nrm1(size_t n, T *x)
{
  cudaStream_t stream{nullptr};
  T result = thrust::reduce(rmm::exec_policy(stream)->on(stream),
                            thrust::device_pointer_cast(x),
                            thrust::device_pointer_cast(x + n));
  CHECK_CUDA(stream);
  return result;
}

template <typename T>
void scal(size_t n, T val, T *x)
{
  cudaStream_t stream{nullptr};
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::device_pointer_cast(x),
                    thrust::device_pointer_cast(x + n),
                    thrust::make_constant_iterator(val),
                    thrust::device_pointer_cast(x),
                    thrust::multiplies<T>());
  CHECK_CUDA(stream);
}

template <typename T>
void addv(size_t n, T val, T *x)
{
  cudaStream_t stream{nullptr};
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::device_pointer_cast(x),
                    thrust::device_pointer_cast(x + n),
                    thrust::make_constant_iterator(val),
                    thrust::device_pointer_cast(x),
                    thrust::plus<T>());
  CHECK_CUDA(stream);
}

template <typename T>
void fill(size_t n, T *x, T value)
{
  cudaStream_t stream{nullptr};
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               thrust::device_pointer_cast(x),
               thrust::device_pointer_cast(x + n),
               value);
  CHECK_CUDA(stream);
}

template <typename T, typename M>
void scatter(size_t n, T *src, T *dst, M *map)
{
  cudaStream_t stream{nullptr};
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
                  thrust::device_pointer_cast(src),
                  thrust::device_pointer_cast(src + n),
                  thrust::device_pointer_cast(map),
                  thrust::device_pointer_cast(dst));
  CHECK_CUDA(stream);
}

template <typename T>
void printv(size_t n, T *vec, int offset)
{
  thrust::device_ptr<T> dev_ptr(vec);
  std::cout.precision(15);
  std::cout << "sample size = " << n << ", offset = " << offset << std::endl;
  thrust::copy(
    dev_ptr + offset,
    dev_ptr + offset + n,
    std::ostream_iterator<T>(
      std::cout, " "));  // Assume no RMM dependency; TODO: check / test (potential BUG !!!!!)
  CHECK_CUDA(nullptr);
  std::cout << std::endl;
}

template <typename T>
void copy(size_t n, T *x, T *res)
{
  thrust::device_ptr<T> dev_ptr(x);
  thrust::device_ptr<T> res_ptr(res);
  cudaStream_t stream{nullptr};
  thrust::copy_n(rmm::exec_policy(stream)->on(stream), dev_ptr, n, res_ptr);
  CHECK_CUDA(stream);
}

template <typename T>
struct is_zero {
  __host__ __device__ bool operator()(const T x) { return x == 0; }
};

template <typename T>
struct dangling_functor : public thrust::unary_function<T, T> {
  const T val;
  dangling_functor(T _val) : val(_val) {}
  __host__ __device__ T operator()(const T &x) const { return val + x; }
};

template <typename T>
void update_dangling_nodes(size_t n, T *dangling_nodes, T damping_factor)
{
  cudaStream_t stream{nullptr};
  thrust::transform_if(rmm::exec_policy(stream)->on(stream),
                       thrust::device_pointer_cast(dangling_nodes),
                       thrust::device_pointer_cast(dangling_nodes + n),
                       thrust::device_pointer_cast(dangling_nodes),
                       dangling_functor<T>(1.0 - damping_factor),
                       is_zero<T>());
  CHECK_CUDA(stream);
}

// google matrix kernels
template <typename IndexType, typename ValueType>
__global__ void degree_coo(const IndexType n,
                           const IndexType e,
                           const IndexType *ind,
                           ValueType *degree)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
    atomicAdd(&degree[ind[i]], (ValueType)1.0);
}

template <typename IndexType, typename ValueType>
__global__ void flag_leafs_kernel(const size_t n, const IndexType *degree, ValueType *bookmark)
{
  for (auto i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    if (degree[i] == 0) bookmark[i] = 1.0;
}

template <typename IndexType, typename ValueType>
__global__ void degree_offsets(const IndexType n,
                               const IndexType e,
                               const IndexType *ind,
                               ValueType *degree)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    degree[i] += ind[i + 1] - ind[i];
}

template <typename FromType, typename ToType>
__global__ void type_convert(FromType *array, int n)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    ToType val   = array[i];
    ToType *vals = (ToType *)array;
    vals[i]      = val;
  }
}

template <typename IndexType, typename ValueType>
__global__ void equi_prob3(const IndexType n,
                           const IndexType e,
                           const IndexType *csrPtr,
                           const IndexType *csrInd,
                           ValueType *val,
                           IndexType *degree)
{
  int j, row, col;
  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
         j += gridDim.y * blockDim.y) {
      col    = csrInd[j];
      val[j] = 1.0 / degree[col];
      // val[j] = 999;
    }
  }
}

template <typename IndexType, typename ValueType>
__global__ void equi_prob2(const IndexType n,
                           const IndexType e,
                           const IndexType *csrPtr,
                           const IndexType *csrInd,
                           ValueType *val,
                           IndexType *degree)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    int row_begin = csrPtr[row];
    int row_end   = csrPtr[row + 1];
    int col;
    for (int i = row_begin; i < row_end; i++) {
      col    = csrInd[i];
      val[i] = 1.0 / degree[col];
    }
  }
}

// compute the H^T values for an already transposed adjacency matrix, leveraging coo info
template <typename IndexType, typename ValueType>
void HT_matrix_csc_coo(const IndexType n,
                       const IndexType e,
                       const IndexType *csrPtr,
                       const IndexType *csrInd,
                       ValueType *val,
                       ValueType *bookmark)
{
  cudaStream_t stream{nullptr};
  rmm::device_vector<IndexType> degree(n, 0);

  dim3 nthreads, nblocks;
  nthreads.x = min(e, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;
  degree_coo<IndexType, IndexType>
    <<<nblocks, nthreads, 0, stream>>>(n, e, csrInd, degree.data().get());
  CHECK_CUDA(stream);

  int y      = 4;
  nthreads.x = 32 / y;
  nthreads.y = y;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, CUDA_MAX_BLOCKS);  // 1;
  equi_prob3<IndexType, ValueType>
    <<<nblocks, nthreads, 0, stream>>>(n, e, csrPtr, csrInd, val, degree.data().get());
  CHECK_CUDA(stream);

  ValueType a = 0.0;
  fill(n, bookmark, a);
  CHECK_CUDA(stream);

  nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;
  flag_leafs_kernel<IndexType, ValueType>
    <<<nblocks, nthreads, 0, stream>>>(n, degree.data().get(), bookmark);
  CHECK_CUDA(stream);
}

template <typename IndexType, typename ValueType>
__global__ void permute_vals_kernel(const IndexType e,
                                    IndexType *perm,
                                    ValueType *in,
                                    ValueType *out)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
    out[i] = in[perm[i]];
}

template <typename IndexType, typename ValueType>
void permute_vals(
  const IndexType e, IndexType *perm, ValueType *in, ValueType *out, cudaStream_t stream = nullptr)
{
  int nthreads = min(e, CUDA_MAX_KERNEL_THREADS);
  int nblocks  = min((e + nthreads - 1) / nthreads, CUDA_MAX_BLOCKS);
  permute_vals_kernel<<<nblocks, nthreads, 0, stream>>>(e, perm, in, out);
}

// This will remove duplicate along with sorting
// This will sort the COO Matrix, row will be sorted and each column of same row will be sorted.
template <typename IndexType, typename ValueType, typename SizeT>
void remove_duplicate(
  IndexType *src, IndexType *dest, ValueType *val, SizeT &nnz, cudaStream_t stream = nullptr)
{
  if (val != NULL) {
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                               thrust::raw_pointer_cast(val),
                               thrust::raw_pointer_cast(val) + nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(
                                 thrust::raw_pointer_cast(src), thrust::raw_pointer_cast(dest))));
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                               thrust::raw_pointer_cast(dest),
                               thrust::raw_pointer_cast(dest + nnz),
                               thrust::make_zip_iterator(thrust::make_tuple(
                                 thrust::raw_pointer_cast(src), thrust::raw_pointer_cast(val))));
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                               thrust::raw_pointer_cast(src),
                               thrust::raw_pointer_cast(src + nnz),
                               thrust::make_zip_iterator(thrust::make_tuple(
                                 thrust::raw_pointer_cast(dest), thrust::raw_pointer_cast(val))));

    typedef thrust::tuple<IndexType *, ValueType *> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::tuple<IndexType *, ZipIterator> ZipIteratorTuple;
    typedef thrust::zip_iterator<ZipIteratorTuple> ZipZipIterator;

    ZipZipIterator newEnd =
      thrust::unique(rmm::exec_policy(stream)->on(stream),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::raw_pointer_cast(src),
                       thrust::make_zip_iterator(thrust::make_tuple(
                         thrust::raw_pointer_cast(dest), thrust::raw_pointer_cast(val))))),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::raw_pointer_cast(src + nnz),
                       thrust::make_zip_iterator(thrust::make_tuple(dest + nnz, val + nnz)))));

    ZipIteratorTuple endTuple = newEnd.get_iterator_tuple();
    IndexType *row_end        = thrust::get<0>(endTuple);

    nnz = ((size_t)row_end - (size_t)src) / sizeof(IndexType);
  } else {
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                               thrust::raw_pointer_cast(dest),
                               thrust::raw_pointer_cast(dest + nnz),
                               thrust::raw_pointer_cast(src));
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                               thrust::raw_pointer_cast(src),
                               thrust::raw_pointer_cast(src + nnz),
                               thrust::raw_pointer_cast(dest));

    typedef thrust::tuple<IndexType *, IndexType *> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    ZipIterator newEnd =
      thrust::unique(rmm::exec_policy(stream)->on(stream),
                     thrust::make_zip_iterator(thrust::make_tuple(thrust::raw_pointer_cast(src),
                                                                  thrust::raw_pointer_cast(dest))),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::raw_pointer_cast(src + nnz), thrust::raw_pointer_cast(dest + nnz))));

    IteratorTuple endTuple = newEnd.get_iterator_tuple();
    IndexType *row_end     = thrust::get<0>(endTuple);

    nnz = ((size_t)row_end - (size_t)src) / sizeof(IndexType);
  }
}

template <typename offsets_t, typename index_t>
__global__ void offsets_to_indices_kernel(const offsets_t *offsets, index_t v, index_t *indices)
{
  auto tid{threadIdx.x};
  auto ctaStart{blockIdx.x};

  for (index_t j = ctaStart; j < v; j += gridDim.x) {
    offsets_t colStart = offsets[j];
    offsets_t colEnd   = offsets[j + 1];
    offsets_t rowNnz   = colEnd - colStart;

    for (offsets_t i = 0; i < rowNnz; i += blockDim.x) {
      if ((colStart + tid + i) < colEnd) { indices[colStart + tid + i] = j; }
    }
  }
}

template <typename offsets_t, typename index_t>
void offsets_to_indices(const offsets_t *offsets, index_t v, index_t *indices)
{
  cudaStream_t stream{nullptr};
  index_t nthreads = min(v, (index_t)CUDA_MAX_KERNEL_THREADS);
  index_t nblocks  = min((v + nthreads - 1) / nthreads, (index_t)CUDA_MAX_BLOCKS);
  offsets_to_indices_kernel<<<nblocks, nthreads, 0, stream>>>(offsets, v, indices);
  CHECK_CUDA(stream);
}

template <typename IndexType>
void sequence(IndexType n, IndexType *vec, IndexType init = 0)
{
  thrust::sequence(
    thrust::device, thrust::device_pointer_cast(vec), thrust::device_pointer_cast(vec + n), init);
  CHECK_CUDA(nullptr);
}

template <typename DistType>
bool has_negative_val(DistType *arr, size_t n)
{
  // custom kernel with boolean bitwise reduce may be
  // faster.
  cudaStream_t stream{nullptr};
  DistType result = *thrust::min_element(rmm::exec_policy(stream)->on(stream),
                                         thrust::device_pointer_cast(arr),
                                         thrust::device_pointer_cast(arr + n));

  CHECK_CUDA(stream);

  return (result < 0);
}

}  // namespace detail
}  // namespace cugraph

#include "eidecl_graph_utils.hpp"

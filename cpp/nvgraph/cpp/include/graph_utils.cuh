/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
// Helper functions based on Thrust


#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
//#include <library_types.h>
//#include <cuda_fp16.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#define USE_CG 1
#define DEBUG 1

namespace nvlouvain
{

#define CUDA_MAX_BLOCKS 65535
#define CUDA_MAX_KERNEL_THREADS 256  //kernel will launch at most 256 threads per block
#define DEFAULT_MASK 0xffffffff
#define US

//#define DEBUG 1

//error check
#undef cudaCheckError 
#ifdef DEBUG
  #define WHERE " at: " << __FILE__ << ':' << __LINE__
  #define cudaCheckError() {                                              \
    cudaError_t e=cudaGetLastError();                                     \
    if(e!=cudaSuccess) {                                                  \
      std::cerr << "Cuda failure: "  << cudaGetErrorString(e) << WHERE << std::endl;        \
    }                                                                     \
  }
#else 
  #define cudaCheckError()   
  #define WHERE ""
#endif 

template<typename T>
static __device__ __forceinline__ T shfl_up(T r, int offset, int bound = 32, int mask = DEFAULT_MASK)
{
    #if __CUDA_ARCH__ >= 300
    #if USE_CG
      return __shfl_up_sync( mask, r, offset, bound );
    #else
      return __shfl_up( r, offset, bound );
    #endif
    #else
      return 0.0f;
    #endif
}

template<typename T>
static __device__ __forceinline__ T shfl(T r, int lane, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#if USE_CG
        return __shfl_sync(mask, r, lane, bound );
#else
        return __shfl(r, lane, bound );
#endif
    #else
        return 0.0f;
    #endif
    }

template<typename T>
__inline__  __device__
T parallel_prefix_sum(int n, int *ind,T *w) {
    int i,j,mn;
    T v,last;
    T sum=0.0;
    bool valid;

    //Parallel prefix sum (using __shfl)
    mn =(((n+blockDim.x-1)/blockDim.x)*blockDim.x); //n in multiple of blockDim.x
    for (i=threadIdx.x; i<mn; i+=blockDim.x) {
        //All threads (especially the last one) must always participate
        //in the shfl instruction, otherwise their sum will be undefined.
        //So, the loop stopping condition is based on multiple of n in loop increments,
        //so that all threads enter into the loop and inside we make sure we do not
        //read out of bounds memory checking for the actual size n.

        //check if the thread is valid
        valid  = i<n;

        //Notice that the last thread is used to propagate the prefix sum.
        //For all the threads, in the first iteration the last is 0, in the following
        //iterations it is the value at the last thread of the previous iterations.

        //get the value of the last thread
        last = shfl(sum, blockDim.x-1, blockDim.x);

        //if you are valid read the value from memory, otherwise set your value to 0
        sum = (valid) ? w[ind[i]] : 0.0;

        //do prefix sum (of size warpSize=blockDim.x =< 32)
        for (j=1; j<blockDim.x; j*=2) {
            v = shfl_up(sum, j, blockDim.x);
            if (threadIdx.x >= j) sum+=v;
        }
        //shift by last
        sum+=last;
        //notice that no __threadfence or __syncthreads are needed in this implementation
    }
    //get the value of the last thread (to all threads)
    last = shfl(sum, blockDim.x-1, blockDim.x);

    return last;
}

//dot
template <typename T>
T dot(size_t n, T* x, T* y) {
  T result = thrust::inner_product(thrust::device_pointer_cast(x), 
                                               thrust::device_pointer_cast(x+n),
                                               thrust::device_pointer_cast(y), 
                                               0.0f);
  cudaCheckError();
  return result;
}

//axpy
template <typename T>
struct axpy_functor : public thrust::binary_function<T,T,T> {
  const T a;
 axpy_functor(T _a) : a(_a) {}
  __host__ __device__
  T operator()(const T& x, const T& y) const { 
      return a * x + y;
    }
};

template <typename T>
void axpy(size_t n, T a,  T* x,  T* y) {
  thrust::transform(thrust::device_pointer_cast(x), 
                              thrust::device_pointer_cast(x+n), 
                              thrust::device_pointer_cast(y), 
                              thrust::device_pointer_cast(y), 
                              axpy_functor<T>(a));
  cudaCheckError();
}

//norm
template <typename T>
struct square {
  __host__ __device__
    T operator()(const T& x) const { 
      return x * x;
    }
};

template <typename T>
T nrm2(size_t n, T* x) {
  T init = 0;
  T result = std::sqrt( thrust::transform_reduce(thrust::device_pointer_cast(x), 
                            thrust::device_pointer_cast(x+n), 
                            square<T>(), 
                            init, 
                            thrust::plus<T>()) );
  cudaCheckError();
  return result;
}

template <typename T>
T nrm1(size_t n, T* x) {
    T result = thrust::reduce(thrust::device_pointer_cast(x), thrust::device_pointer_cast(x+n));
    cudaCheckError();
    return result;
}

template <typename T>
void scal(size_t n, T val, T* x) {
  thrust::transform(thrust::device_pointer_cast(x),
                                  thrust::device_pointer_cast(x + n),  
                                  thrust::make_constant_iterator(val), 
                                  thrust::device_pointer_cast(x), 
                                  thrust::multiplies<T>());
  cudaCheckError();
}

template <typename T>
void fill(size_t n, T* x, T value) {
    thrust::fill(thrust::device_pointer_cast(x), thrust::device_pointer_cast(x + n), value);
    cudaCheckError();
}

template <typename T>
void printv(size_t n, T* vec, int offset) {
    thrust::device_ptr<T> dev_ptr(vec);
    std::cout.precision(15);
    std::cout << "sample size = "<< n << ", offset = "<< offset << std::endl;
    thrust::copy(dev_ptr+offset,dev_ptr+offset+n, std::ostream_iterator<T>(std::cout, " "));
    cudaCheckError();
    std::cout << std::endl;
}

template<typename T>
void copy(size_t n, T *x, T *res)
{
    thrust::device_ptr<T> dev_ptr(x);
    thrust::device_ptr<T> res_ptr(res);
    thrust::copy_n(dev_ptr, n, res_ptr);
    cudaCheckError();
}

template <typename T>
struct is_zero {
  __host__ __device__
  bool operator()(const T x) {
    return x == 0;
  }
};

template <typename T>
struct dangling_functor : public thrust::unary_function<T,T> {
  const T val;
  dangling_functor(T _val) : val(_val) {}
  __host__ __device__
  T operator()(const T& x) const { 
      return val + x;
    }
};

template <typename T>
void update_dangling_nodes(size_t n, T* dangling_nodes, T damping_factor) {
  thrust::transform_if(thrust::device_pointer_cast(dangling_nodes),
  	thrust::device_pointer_cast( dangling_nodes + n),  
  	thrust::device_pointer_cast(dangling_nodes), 
  	dangling_functor<T>(1.0-damping_factor),
  	is_zero<T>());
  cudaCheckError();
}

//google matrix kernels
template <typename IndexType, typename ValueType>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
degree_coo ( const  IndexType n, const IndexType e, const IndexType *ind, IndexType *degree) {
    for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<e; i+=gridDim.x*blockDim.x) 
        atomicAdd(&degree[ind[i]],1.0);
}
template <typename IndexType, typename ValueType>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
equi_prob ( const  IndexType n, const IndexType e, const IndexType *ind, ValueType *val, IndexType *degree) {
    for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<e; i+=gridDim.x*blockDim.x) 
        val[i] = 1.0/degree[ind[i]];
}

template <typename IndexType, typename ValueType>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
flag_leafs ( const  IndexType n, IndexType *degree, ValueType *bookmark) {
    for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) 
      if (degree[i]==0)
        bookmark[i]=1.0;
}
//notice that in the transposed matrix/csc a dangling node is a node without incomming edges
template <typename IndexType, typename ValueType>
void google_matrix ( const  IndexType n, const IndexType e, const IndexType *cooColInd, ValueType *cooVal, ValueType *bookmark) {
  thrust::device_vector<IndexType> degree(n,0);
  dim3 nthreads, nblocks;
  nthreads.x = min(e,CUDA_MAX_KERNEL_THREADS); 
  nthreads.y = 1; 
  nthreads.z = 1;  
  nblocks.x  = min((e + nthreads.x - 1)/nthreads.x,CUDA_MAX_BLOCKS); 
  nblocks.y  = 1; 
  nblocks.z  = 1;
  degree_coo<IndexType,ValueType><<<nblocks,nthreads>>>(n,e,cooColInd, thrust::raw_pointer_cast(degree.data()));
  equi_prob<IndexType,ValueType><<<nblocks,nthreads>>>(n,e,cooColInd, cooVal, thrust::raw_pointer_cast(degree.data()));
  ValueType val = 0.0;
  fill(n,bookmark,val);
  nthreads.x = min(n,CUDA_MAX_KERNEL_THREADS); 
  nblocks.x  = min((n + nthreads.x - 1)/nthreads.x,CUDA_MAX_BLOCKS); 
  flag_leafs <IndexType,ValueType><<<nblocks,nthreads>>>(n, thrust::raw_pointer_cast(degree.data()), bookmark);
  //printv(n, thrust::raw_pointer_cast(degree.data()) , 0);
  //printv(n, bookmark , 0);
  //printv(e, cooVal , 0);
}

template <typename IndexType>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
update_clustering_kernel ( const  IndexType n, IndexType *clustering, IndexType *aggregates_d) {
    for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) 
      clustering[i] = aggregates_d[clustering[i]];
}

template <typename IndexType>
void update_clustering ( const  IndexType n, IndexType *clustering, IndexType *aggregates_d) {
  int nthreads = min(n,CUDA_MAX_KERNEL_THREADS); 
  int nblocks = min((n + nthreads - 1)/nthreads,CUDA_MAX_BLOCKS); 
  update_clustering_kernel<IndexType><<<nblocks,nthreads>>>(n,clustering,aggregates_d);
}

} //namespace nvga

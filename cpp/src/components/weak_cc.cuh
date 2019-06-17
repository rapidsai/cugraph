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
#pragma once

///#include "cuda_utils.h"
///#include "array/array.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <type_traits>

#include "utils.h"


namespace MLCommon {

  
/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType1,
          typename IntType2>
constexpr inline __host__ __device__
IntType1 ceildiv(IntType1 a, IntType2 b) {
  return (a + b - 1) / b;
}

template<typename T, typename IndexT = int>
__device__ IndexT get_stop_idx(T row, IndexT m, IndexT nnz, const T *ind) {
    IndexT stop_idx = 0;
    if(row < (m-1))
        stop_idx = ind[row+1];
    else
        stop_idx = nnz;

    return stop_idx;
}
  
namespace Sparse {

template<typename T = int>
class WeakCCState {
public:
  //using bool_ = char;
  
  bool *xa;
  bool *fa;
  bool *m;
  bool owner;

  WeakCCState(T n):
    xa(nullptr),
    fa(nullptr),
    m(nullptr),
    owner(true)
  {
    MLCommon::allocate(xa, n, true);
    MLCommon::allocate(fa, n, true);
    MLCommon::allocate(m, 1, true);
    // h_p_d_xa = new thrust::device_vector<bool>(n, 1);
    // h_p_d_fa = new thrust::device_vector<bool>(n, 1);
    // h_p_d_m = new thrust::device_vector<bool>(1, 1);

    // xa = h_p_d_xa->data().get();
    // fa = h_p_d_fa->data().get();
    // m  = h_p_d_m->data().get();
  }

  WeakCCState(bool *xa, bool *fa, bool *m):
    owner(false), xa(xa), fa(fa), m(m) {
  }

  ~WeakCCState() {
    if(owner) {
      try {
        // delete h_p_d_xa;
        // delete h_p_d_fa;
        // delete h_p_d_m;
               
        cudaStream_t stream{nullptr};
        
        if( xa )
          ALLOC_FREE_TRY(xa, stream);
          //CUDA_CHECK(cudaFree(xa));

        if( fa )
          ALLOC_FREE_TRY(fa, stream);
          //CUDA_CHECK(cudaFree(fa));

        if( m )
          ALLOC_FREE_TRY(m,  stream);
          //CUDA_CHECK(cudaFree(m));

        xa = nullptr;
        fa = nullptr;
        m = nullptr;
          
        
      } catch(Exception &e) {
        std::cout << "Exception freeing memory for WeakCCState: " <<
          e.what() << std::endl;
      }
    }
  }
// private:
//   thrust::device_vector<bool>* h_p_d_xa;
//   thrust::device_vector<bool>* h_p_d_fa;
//   thrust::device_vector<bool>* h_p_d_m;
};

template <typename Type, int TPB_X = 32>
__global__ void weak_cc_label_device(
        Type *labels,
        const Type *row_ind, const Type *row_ind_ptr, Type nnz,
        bool *fa, bool *xa, bool *m,
        Type startVertexId, Type batchSize) {
    Type tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<batchSize) {
        if(fa[tid + startVertexId]) {
            fa[tid + startVertexId] = false;
            Type start = row_ind[tid];
            Type ci, cj;
            bool ci_mod = false;
            ci = labels[tid + startVertexId];

            Type degree = get_stop_idx(tid, batchSize,nnz, row_ind) - row_ind[tid];

            for(auto j=0; j< degree; j++) { // TODO: Can't this be calculated from the ex_scan?
                cj = labels[row_ind_ptr[start + j]];
                if(ci<cj) {
                    atomicMin(labels + row_ind_ptr[start +j], ci);
                    xa[row_ind_ptr[start+j]] = true;
                    m[0] = true;
                }
                else if(ci>cj) {
                    ci = cj;
                    ci_mod = true;
                }
            }
            if(ci_mod) {
                atomicMin(labels + startVertexId + tid, ci);
                xa[startVertexId + tid] = true;
                m[0] = true;
            }
        }
    }
}


template <typename Type, int TPB_X = 32, typename Lambda>
__global__ void weak_cc_init_label_kernel(Type *labels, Type startVertexId, Type batchSize,
        Type MAX_LABEL, Lambda filter_op) {
    /** F1 and F2 in the paper correspond to fa and xa */
    /** Cd in paper corresponds to db_cluster */
    Type tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<batchSize) {
        if(filter_op(tid) && labels[tid + startVertexId]==MAX_LABEL)
            labels[startVertexId + tid] = Type(startVertexId + tid + 1);
    }
}

template <typename Type, int TPB_X = 32>
__global__ void weak_cc_init_all_kernel(Type *labels, bool *fa, bool *xa,
        Type N, Type MAX_LABEL) {
    Type tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<N) {
        labels[tid] = MAX_LABEL;
        fa[tid] = true;
        xa[tid] = false;
    }
}

  template <typename Type, int TPB_X = 32, typename Lambda>
void weak_cc_label_batched(Type *labels,
        const Type* row_ind, const Type* row_ind_ptr, Type nnz, Type N,
        WeakCCState<Type> *state,
        Type startVertexId, Type batchSize,
        cudaStream_t stream, Lambda filter_op) {
    bool host_m;
    bool *host_fa = (bool*)malloc(sizeof(bool)*N);
    bool *host_xa = (bool*)malloc(sizeof(bool)*N);

    dim3 blocks(ceildiv(batchSize, TPB_X));
    dim3 threads(TPB_X);
    Type MAX_LABEL = std::numeric_limits<Type>::max();

    weak_cc_init_label_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>(labels,
            startVertexId, batchSize, MAX_LABEL, filter_op);
    CUDA_CHECK(cudaPeekAtLastError());
    do {
        CUDA_CHECK( cudaMemsetAsync(state->m, false, sizeof(bool), stream) );
        weak_cc_label_device<Type, TPB_X><<<blocks, threads, 0, stream>>>(
                labels,
                row_ind, row_ind_ptr, nnz,
                state->fa, state->xa, state->m,
                startVertexId, batchSize);
        CUDA_CHECK(cudaPeekAtLastError());

        //** swapping F1 and F2
        MLCommon::updateHost(host_fa, state->fa, N, stream);
        MLCommon::updateHost(host_xa, state->xa, N, stream);
        MLCommon::updateDevice(state->fa, host_xa, N, stream);
        MLCommon::updateDevice(state->xa, host_fa, N, stream);

        //** Updating m *
        MLCommon::updateHost(&host_m, state->m, 1, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } while(host_m);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling.
 */
template<typename Type = int, int TPB_X = 32, typename Lambda = auto (Type)->bool>
void weak_cc_batched(Type *labels, const Type* row_ind, const Type* row_ind_ptr,
        Type nnz, Type N, Type startVertexId, Type batchSize,
        WeakCCState<Type> *state, cudaStream_t stream, Lambda filter_op) {

    dim3 blocks(ceildiv(N, TPB_X));
    dim3 threads(TPB_X);

    Type MAX_LABEL = std::numeric_limits<Type>::max();
    if(startVertexId == 0) {
        weak_cc_init_all_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>
            (labels, state->fa, state->xa, N, MAX_LABEL);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    weak_cc_label_batched<Type, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, state,
            startVertexId, batchSize, stream, filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 */
template<typename Type = int, int TPB_X = 32>
void weak_cc_batched(Type *labels, const Type* row_ind,  const Type* row_ind_ptr,
        Type nnz, Type N, Type startVertexId, Type batchSize,
        WeakCCState<Type> *state, cudaStream_t stream) {

    weak_cc_batched(labels, row_ind, row_ind_ptr, nnz, N, startVertexId, batchSize,
            state, stream, [] __device__ (Type tid) {return true;});
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling.
 */
template<typename Type = int, int TPB_X = 32, typename Lambda = auto (Type)->bool>
void weak_cc(Type *labels, const Type* row_ind, const Type* row_ind_ptr,
        Type nnz, Type N, cudaStream_t stream, Lambda filter_op) {

    WeakCCState<Type> state(N);
    weak_cc_batched<Type, TPB_X>(
            labels, row_ind, row_ind_ptr,
            nnz, N, 0, N, stream,
            filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param stream the cuda stream to use
 * should get considered for labeling.
 */
template<typename Type = int, int TPB_X = 32>
void weak_cc_entry(Type *labels,
                   const Type* row_ind,
                   const Type* row_ind_ptr,
                   Type nnz,
                   Type N,
                   cudaStream_t stream) {

  WeakCCState<Type> state(N);
  //WeakCCState<Type>* p_state(new WeakCCState<Type>(N));//leak memory on purpose...
  weak_cc_batched<Type, TPB_X>(labels, row_ind, row_ind_ptr,
                               nnz, N, 0, N, &state, stream);
  ///[] __device__ (Type t){return true;});//this works, but subject to deprecation
  
  cudaDeviceSynchronize();
}
  
}
}

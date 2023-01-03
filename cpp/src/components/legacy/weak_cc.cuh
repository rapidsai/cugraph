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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/swap.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <type_traits>

#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>

#include "utils.h"
#include <rmm/device_vector.hpp>

namespace MLCommon {

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType1, typename IntType2>
constexpr inline __host__ __device__ IntType1 ceildiv(IntType1 a, IntType2 b)
{
  return (a + b - 1) / b;
}

namespace Sparse {

class WeakCCState {
 public:
  bool* xa;
  bool* fa;
  bool* m;
  bool owner;

  WeakCCState(bool* xa, bool* fa, bool* m) : xa(xa), fa(fa), m(m) {}
};

template <typename vertex_t, typename edge_t, int TPB_X = 32>
__global__ void weak_cc_label_device(vertex_t* labels,
                                     edge_t const* offsets,
                                     vertex_t const* indices,
                                     edge_t nnz,
                                     bool* fa,
                                     bool* xa,
                                     bool* m,
                                     vertex_t startVertexId,
                                     vertex_t batchSize)
{
  vertex_t tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < batchSize) {
    if (fa[tid + startVertexId]) {
      fa[tid + startVertexId] = false;
      vertex_t ci, cj;
      bool ci_mod = false;
      ci          = labels[tid + startVertexId];

      // TODO:
      //    This can't be optimal.  A high degree vertex will cause
      //    terrible load balancing, since one thread on the GPU will
      //    have to do all of that work.
      //

      //
      // NOTE: reworked this loop.  In cugraph offsets is one element
      //       longer so you can always do this memory reference.
      //
      //  edge_t degree = get_stop_idx(tid, batchSize, nnz, offsets) - offsets[tid];
      //
      // edge_t degree = offsets[tid+1] - offsets[tid];
      // for (auto j = 0 ; j < degree ; j++) { // TODO: Can't this be calculated from the ex_scan?
      //  vertex_t j_ind = indices[start+j];
      //  ...
      // }
      //
      for (edge_t j = offsets[tid]; j < offsets[tid + 1]; ++j) {
        vertex_t j_ind = indices[j];
        cj             = labels[j_ind];
        if (ci < cj) {
          atomicMin(labels + j_ind, ci);
          xa[j_ind] = true;
          m[0]      = true;
        } else if (ci > cj) {
          ci     = cj;
          ci_mod = true;
        }
      }

      if (ci_mod) {
        atomicMin(labels + startVertexId + tid, ci);
        xa[startVertexId + tid] = true;
        m[0]                    = true;
      }
    }
  }
}

template <typename vertex_t, int TPB_X = 32, typename Lambda>
__global__ void weak_cc_init_label_kernel(vertex_t* labels,
                                          vertex_t startVertexId,
                                          vertex_t batchSize,
                                          vertex_t MAX_LABEL,
                                          Lambda filter_op)
{
  /** F1 and F2 in the paper correspond to fa and xa */
  /** Cd in paper corresponds to db_cluster */
  vertex_t tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < batchSize) {
    if (filter_op(tid) && labels[tid + startVertexId] == MAX_LABEL)
      labels[startVertexId + tid] = vertex_t{startVertexId + tid + 1};
  }
}

template <typename vertex_t, int TPB_X = 32>
__global__ void weak_cc_init_all_kernel(
  vertex_t* labels, bool* fa, bool* xa, vertex_t N, vertex_t MAX_LABEL)
{
  vertex_t tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    labels[tid] = MAX_LABEL;
    fa[tid]     = true;
    xa[tid]     = false;
  }
}

template <typename vertex_t, typename edge_t, int TPB_X = 32, typename Lambda>
void weak_cc_label_batched(vertex_t* labels,
                           edge_t const* offsets,
                           vertex_t const* indices,
                           edge_t nnz,
                           vertex_t N,
                           WeakCCState& state,
                           vertex_t startVertexId,
                           vertex_t batchSize,
                           cudaStream_t stream,
                           Lambda filter_op)
{
  ASSERT(sizeof(vertex_t) == 4 || sizeof(vertex_t) == 8, "Index_ should be 4 or 8 bytes");

  bool host_m{true};

  dim3 blocks(ceildiv(batchSize, vertex_t{TPB_X}));
  dim3 threads(TPB_X);
  vertex_t MAX_LABEL = std::numeric_limits<vertex_t>::max();

  weak_cc_init_label_kernel<vertex_t, TPB_X>
    <<<blocks, threads, 0, stream>>>(labels, startVertexId, batchSize, MAX_LABEL, filter_op);

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  int n_iters = 0;
  do {
    RAFT_CUDA_TRY(cudaMemsetAsync(state.m, false, sizeof(bool), stream));

    weak_cc_label_device<vertex_t, edge_t, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, offsets, indices, nnz, state.fa, state.xa, state.m, startVertexId, batchSize);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    thrust::swap(state.fa, state.xa);

    //** Updating m *
    MLCommon::updateHost(&host_m, state.m, 1, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    n_iters++;
  } while (host_m);
}

/**
 * @brief Compute weakly connected components.
 *
 * Note that the resulting labels may not be taken from a monotonically
 * increasing set (eg. numbers may be skipped). The MLCommon::Array
 * package contains a primitive `make_monotonic`, which will make a
 * monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam vertex_t   The type of a vertex id
 * @tparam edge_t     The type of an edge id
 * @tparam TPB_X      Number of threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 *
 * @param labels      Array for the output labels
 * @param offsets     CSR offsets array
 * @param indices     CSR indices array
 * @param nnz         Number of edges
 * @param N           Number of vertices
 * @param stream      Cuda stream to use
 * @param filter_op   Optional filtering function to determine which points
 *                    should get considered for labeling.
 */
template <typename vertex_t,
          typename edge_t,
          int TPB_X       = 32,
          typename Lambda = auto(vertex_t)->bool>
void weak_cc_batched(vertex_t* labels,
                     edge_t const* offsets,
                     vertex_t const* indices,
                     edge_t nnz,
                     vertex_t N,
                     vertex_t startVertexId,
                     vertex_t batchSize,
                     WeakCCState& state,
                     cudaStream_t stream,
                     Lambda filter_op)
{
  dim3 blocks(ceildiv(N, TPB_X));
  dim3 threads(TPB_X);

  vertex_t MAX_LABEL = std::numeric_limits<vertex_t>::max();
  if (startVertexId == 0) {
    weak_cc_init_all_kernel<vertex_t, TPB_X>
      <<<blocks, threads, 0, stream>>>(labels, state.fa, state.xa, N, MAX_LABEL);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  weak_cc_label_batched<vertex_t, edge_t, TPB_X>(
    labels, offsets, indices, nnz, N, state, startVertexId, batchSize, stream, filter_op);
}

/**
 * @brief Compute weakly connected components.
 *
 * Note that the resulting labels may not be taken from a monotonically
 * increasing set (eg. numbers may be skipped). The MLCommon::Array
 * package contains a primitive `make_monotonic`, which will make a
 * monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam vertex_t   The type of a vertex id
 * @tparam edge_t     The type of an edge id
 * @tparam TPB_X      Number of threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 *
 * @param labels      Array for the output labels
 * @param offsets     CSR offsets array
 * @param indices     CSR indices array
 * @param nnz         Number of edges
 * @param N           Number of vertices
 * @param stream      Cuda stream to use
 * @param filter_op   Optional filtering function to determine which points
 *                    should get considered for labeling.
 */
template <typename vertex_t,
          typename edge_t,
          int TPB_X       = 32,
          typename Lambda = auto(vertex_t)->bool>
void weak_cc(vertex_t* labels,
             edge_t const* offsets,
             vertex_t const* indices,
             edge_t nnz,
             vertex_t N,
             cudaStream_t stream,
             Lambda filter_op)
{
  rmm::device_vector<bool> xa(N);
  rmm::device_vector<bool> fa(N);
  rmm::device_vector<bool> m(1);

  WeakCCState state(xa.data().get(), fa.data().get(), m.data().get());
  weak_cc_batched<vertex_t, edge_t, TPB_X>(
    labels, offsets, indices, nnz, N, 0, N, state, stream, filter_op);
}

/**
 * @brief Compute weakly connected components.
 *
 * Note that the resulting labels may not be taken from a monotonically
 * increasing set (eg. numbers may be skipped). The MLCommon::Array
 * package contains a primitive `make_monotonic`, which will make a
 * monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam vertex_t   The type of a vertex id
 * @tparam edge_t     The type of an edge id
 * @tparam TPB_X      Number of threads to use per block when configuring the kernel
 *
 * @param labels      Array for the output labels
 * @param offsets     CSR offsets array
 * @param indices     CSR indices array
 * @param nnz         Number of edges
 * @param N           Number of vertices
 * @param stream      Cuda stream to use
 */
template <typename vertex_t, typename edge_t, int TPB_X = 32>
void weak_cc_entry(vertex_t* labels,
                   edge_t const* offsets,
                   vertex_t const* indices,
                   edge_t nnz,
                   vertex_t N,
                   cudaStream_t stream)
{
  weak_cc(labels, offsets, indices, nnz, N, stream, [] __device__(vertex_t) { return true; });
}

}  // namespace Sparse
}  // namespace MLCommon

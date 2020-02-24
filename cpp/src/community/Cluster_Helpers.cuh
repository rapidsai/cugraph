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

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm_utils.h>
#include <utilities/error_utils.h>

namespace {

template<typename IdxT, typename ValT>
struct level_info {
  IdxT num_verts;
  IdxT nnz;
  IdxT num_clusters;
  IdxT* csr_off;
  IdxT* csr_ind;
  ValT* csr_val;
  IdxT* clusters;
  IdxT* cluster_inv_off;
  IdxT* cluster_inv_ind;

  void delete_all() {
    ALLOC_FREE_TRY(csr_off, nullptr);
    ALLOC_FREE_TRY(csr_ind, nullptr);
    ALLOC_FREE_TRY(csr_val, nullptr);
    ALLOC_FREE_TRY(clusters, nullptr);
    ALLOC_FREE_TRY(cluster_inv_off, nullptr);
    ALLOC_FREE_TRY(cluster_inv_ind, nullptr);
  }

  void delete_added() {
    ALLOC_FREE_TRY(clusters, nullptr);
    ALLOC_FREE_TRY(cluster_inv_off, nullptr);
    ALLOC_FREE_TRY(cluster_inv_ind, nullptr);
  }
};

template<typename ValType, typename IdxType>
__device__ void compute_k_vec(const int n_vertex,
                              IdxType* csr_ptr_ptr,
                              ValType* csr_val_ptr,
                              bool weighted,
                              ValType* k_vec) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if ((tid < n_vertex)) {

    int start_idx = *(csr_ptr_ptr + tid);
    int end_idx = *(csr_ptr_ptr + tid + 1);

    if (!weighted) {
      *(k_vec + tid) = (ValType) end_idx - start_idx;
    }
    else {
      ValType sum = 0.0;
      for (int i = 0; i < end_idx - start_idx; ++i) {
        sum += *(csr_val_ptr + start_idx + i);
      }
      *(k_vec + tid) = sum;
    }
  }
  return;
}

template<typename IdxT, typename ValT>
__global__ void compute_k_vec_kernel(IdxT n_vertex,
                                     IdxT* csr_off,
                                     ValT* csr_val,
                                     ValT* k_vec) {
  compute_k_vec(n_vertex, csr_off, csr_val, true, k_vec);
}

template<typename IdxT, typename ValT>
void compute_k_vec(IdxT n_vertex,
                   IdxT* csr_off,
                   ValT* csr_val,
                   ValT* k_vec) {
  int nthreads = min(n_vertex, (IdxT)CUDA_MAX_KERNEL_THREADS);
  int nblocks = min((n_vertex + nthreads - 1) / nthreads, (IdxT)CUDA_MAX_BLOCKS);
  compute_k_vec_kernel<<<nblocks, nthreads>>>(n_vertex,
                                              csr_off,
                                              csr_val,
                                              k_vec);
  CUDA_TRY(cudaDeviceSynchronize());
}

template<typename IndexType>
void renumberAndCountAggregates(IndexType* aggregates, const IndexType n, IndexType& num_aggregates)
                                {
  // renumber aggregates
  rmm::device_vector<IndexType> scratch(n + 1, 0);
  thrust::device_ptr<IndexType> aggregates_thrust_dev_ptr(aggregates);
  thrust::device_ptr<IndexType> scratch_thrust_dev_ptr(scratch.data());

  // set scratch[aggregates[i]] = 1
  thrust::fill(thrust::make_permutation_iterator(scratch_thrust_dev_ptr, aggregates_thrust_dev_ptr),
               thrust::make_permutation_iterator(scratch_thrust_dev_ptr,
                                                 aggregates_thrust_dev_ptr + n),
               1);

  // do prefix sum on scratch
  thrust::exclusive_scan(scratch_thrust_dev_ptr,
                         scratch_thrust_dev_ptr + n + 1,
                         scratch_thrust_dev_ptr);
  // scratch.dump(0,scratch.get_size());

  // aggregates[i] = scratch[aggregates[i]]
  thrust::copy(thrust::make_permutation_iterator(scratch_thrust_dev_ptr, aggregates_thrust_dev_ptr),
               thrust::make_permutation_iterator(scratch_thrust_dev_ptr,
                                                 aggregates_thrust_dev_ptr + n),
               aggregates_thrust_dev_ptr);
  CUDA_CHECK_LAST();
  cudaMemcpy(&num_aggregates,
             thrust::raw_pointer_cast(scratch.data() + n),
             sizeof(int),
             cudaMemcpyDefault); //num_aggregates = scratch.raw()[scratch.get_size()-1];
  CUDA_CHECK_LAST();
}

} // namespace anonymous

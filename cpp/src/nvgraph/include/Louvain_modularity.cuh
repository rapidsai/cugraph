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

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

namespace nvlouvain {
template<typename IndexType>
__device__ IndexType binsearch_maxle(const IndexType *vec,
                                      const IndexType val,
                                      IndexType low,
                                      IndexType high) {
  while (true) {
    if (low == high)
      return low; //we know it exists
    if ((low + 1) == high)
      return (vec[high] <= val) ? high : low;

    IndexType mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

template<typename IdxT, typename ValT>
__global__ void compute_e_c_vector(IdxT nnz,
                                   IdxT num_verts,
                                   IdxT* csr_off,
                                   IdxT* csr_ind,
                                   ValT* edge_weights,
                                   IdxT* clusters,
                                   IdxT* cluster_id,
                                   ValT* e_c_sum){
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < nnz) {
    IdxT startVertex = binsearch_maxle(csr_off, tid, (IdxT)0, num_verts);
    IdxT endVertex = csr_ind[tid];
    IdxT startCluster = clusters[startVertex];
    IdxT endCluster = clusters[endVertex];
    cluster_id[tid] = startCluster;
    e_c_sum[tid] = (startCluster == endCluster && startVertex != endVertex) ? edge_weights[tid] : (ValT)0.0;

    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
void compute_e_c(IdxT nnz,
                 IdxT num_verts,
                 IdxT* csr_off,
                 IdxT* csr_ind,
                 ValT* edge_weights,
                 IdxT* clusters,
                 ValT* e_c) {
  rmm::device_vector<ValT> e_c_sum(nnz, 0.0);
  rmm::device_vector<IdxT> cluster_id(nnz, 0);
  rmm::device_vector<IdxT> cluster_id_out(nnz,0);
  ValT* e_c_sum_ptr = thrust::raw_pointer_cast(e_c_sum.data());
  IdxT* cluster_id_ptr = thrust::raw_pointer_cast(cluster_id.data());
  IdxT* cluster_id_out_ptr = thrust::raw_pointer_cast(cluster_id_out.data());

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected before compute_e_c_vector: " << cudaGetErrorString(error)
        << "\n";
  }

  dim3 grid, block;
  block.x = 512;
  grid.x = min((IdxT) CUDA_MAX_BLOCKS, (nnz / 512 + 1));
  compute_e_c_vector<<<grid, block, 0, nullptr>>>(nnz,
                                                  num_verts,
                                                  csr_off,
                                                  csr_ind,
                                                  edge_weights,
                                                  clusters,
                                                  cluster_id_ptr,
                                                  e_c_sum_ptr);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected after compute_e_c_vector: " << cudaGetErrorString(error)
        << "\n";
  }

  thrust::sort_by_key(thrust::cuda::par, cluster_id_ptr, cluster_id_ptr + nnz, e_c_sum_ptr);
  thrust::reduce_by_key(thrust::cuda::par,
                        cluster_id_ptr,
                        cluster_id_ptr + nnz,
                        e_c_sum_ptr,
                        cluster_id_out_ptr,
                        e_c);
}

template<typename IdxT, typename ValT>
__global__ void compute_k_c_vector(IdxT num_verts,
                                   IdxT* clusters,
                                   ValT* k_vec,
                                   IdxT* cluster_id,
                                   ValT* k_c) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT clusterId = clusters[tid];
    ValT kVal = k_vec[tid];
    cluster_id[tid] = clusterId;
    k_c[tid] = kVal;
    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
void compute_k_c(IdxT num_verts,
                 IdxT* clusters,
                 ValT* k_vec,
                 ValT* k_c) {
  rmm::device_vector<ValT> k_c_sum(num_verts, 0.0);
  rmm::device_vector<IdxT> cluster_id(num_verts, 0);
  rmm::device_vector<IdxT> cluster_id_out(num_verts, 0);
  ValT* k_c_sum_ptr = thrust::raw_pointer_cast(k_c_sum.data());
  IdxT* cluster_id_ptr = thrust::raw_pointer_cast(cluster_id.data());
  IdxT* cluster_id_out_ptr = thrust::raw_pointer_cast(cluster_id_out.data());

  dim3 grid, block;
  block.x = 512;
  grid.x = min((IdxT) CUDA_MAX_BLOCKS, (num_verts / 512 + 1));
  compute_k_c_vector<<<grid, block, 0, nullptr>>>(num_verts,
                                                  clusters,
                                                  k_vec,
                                                  cluster_id_ptr,
                                                  k_c_sum_ptr);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected after compute_k_c_vector: " << cudaGetErrorString(error)
        << "\n";
  }

  thrust::sort_by_key(thrust::cuda::par, cluster_id_ptr, cluster_id_ptr + num_verts, k_c_sum_ptr);
  thrust::reduce_by_key(thrust::cuda::par,
                        cluster_id_ptr,
                        cluster_id_ptr + num_verts,
                        k_c_sum_ptr,
                        cluster_id_out_ptr,
                        k_c);
}

template<typename IdxT, typename ValT>
__global__ void compute_m_c_vector(IdxT num_clusters,
                                   ValT* e_c,
                                   ValT* k_c,
                                   ValT gamma,
                                   ValT m2,
                                   ValT* m_c) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < num_clusters) {
    ValT kVal = k_c[tid];
    ValT eVal = e_c[tid];
    ValT mVal = (eVal/m2) - gamma * ((kVal * kVal) / m2 / m2);
//    printf("Thread %d, kVal %f, eVal %f, mVal %f\n", tid, kVal, eVal, mVal);
    m_c[tid] = mVal;
    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
ValT compute_modularity(IdxT num_verts,
                        IdxT nnz,
                        IdxT num_clusters,
                        IdxT* csr_off,
                        IdxT* csr_ind,
                        ValT* edge_weights,
                        ValT* k_vec,
                        IdxT* clusters,
                        ValT gamma,
                        ValT m2,
                        ValT* e_c,
                        ValT* k_c,
                        ValT* m_c) {
  compute_e_c(nnz, num_verts, csr_off, csr_ind, edge_weights, clusters, e_c);
  compute_k_c(num_verts, clusters, k_vec, k_c);

  dim3 grid, block;
  block.x = 512;
  grid.x = min((IdxT)CUDA_MAX_BLOCKS, (num_clusters / 512 + 1));

  compute_m_c_vector<<<grid, block, 0, nullptr>>>(num_clusters, e_c, k_c, gamma, m2, m_c);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected after compute_m_c_vector: " << cudaGetErrorString(error)
        << "\n";
  }

  ValT modularity = thrust::reduce(thrust::cuda::par, m_c, m_c + num_clusters);
  return modularity;
}

template<typename IdxT, typename ValT>
__global__ void compute_delta_modularity_vector(IdxT nnz,
                                                IdxT num_verts,
                                                IdxT* csr_off,
                                                IdxT* csr_ind,
                                                ValT* edge_weights,
                                                IdxT* clusters,
                                                ValT* e_c,
                                                ValT* k_c,
                                                ValT* m_c,
                                                ValT* k_vec,
                                                ValT* delta,
                                                ValT gamma,
                                                ValT m2) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < nnz) {
    IdxT startVertex = binsearch_maxle(csr_off, tid, (IdxT)0, num_verts);
    IdxT endVertex = csr_ind[tid];
    IdxT startCluster = clusters[startVertex];
    IdxT endCluster = clusters[endVertex];

    if (startCluster == endCluster) {
      delta[tid] = 0.0;
    }
    else {
      ValT endEdges = 0.0;
      ValT startEdges = 0.0;
      IdxT start = csr_off[startVertex];
      IdxT end = csr_off[startVertex + 1];
      for (IdxT i = start; i < end; i++) {
        IdxT neighborId = csr_ind[i];
        IdxT neighborCluster = clusters[neighborId];
        if (neighborCluster == startCluster && neighborId != startVertex)
          startEdges += edge_weights[i];
        if (neighborCluster == endCluster && neighborId != startVertex)
          endEdges += edge_weights[i];
      }
      ValT end_e = e_c[endCluster];
      ValT end_k = k_c[endCluster];
      ValT my_k = k_vec[startVertex];
      ValT newEndScore = (end_e + endEdges) / m2 - gamma * (((end_k + my_k)*(end_k + my_k)) / m2 / m2);
      ValT start_e = e_c[startCluster];
      ValT start_k = k_c[startCluster];
      ValT newStartScore = (start_e - startEdges) / m2 - gamma * (((start_k - my_k)*(start_k - my_k)) / m2 / m2);
      ValT startM = m_c[startCluster];
      ValT endM = m_c[endCluster];
      ValT finalScore = newEndScore - endM + newStartScore - startM;
//      printf("Thread: %d, finalScore %f, newEndScore %f, oldEndScore %f, newStartScore %f, oldStartScore %f\n", tid, finalScore, newEndScore, endM, newStartScore, startM);
      if (newEndScore < endM)
        finalScore = 0.0;
      delta[tid] = finalScore < .0001 ? 0.0 : finalScore;
    }

    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
void compute_delta_modularity(IdxT nnz,
                              IdxT num_verts,
                              IdxT* csr_off,
                              IdxT* csr_ind,
                              ValT* edge_weights,
                              IdxT* clusters,
                              ValT* e_c,
                              ValT* k_c,
                              ValT* m_c,
                              ValT* k_vec,
                              ValT* delta,
                              ValT gamma,
                              ValT m2){
  dim3 grid, block;
  block.x = 512;
  grid.x = min((IdxT)CUDA_MAX_BLOCKS, (nnz / 512 + 1));
  compute_delta_modularity_vector<<<grid, block, 0, nullptr>>>(nnz,
                                                               num_verts,
                                                               csr_off,
                                                               csr_ind,
                                                               edge_weights,
                                                               clusters,
                                                               e_c,
                                                               k_c,
                                                               m_c,
                                                               k_vec,
                                                               delta,
                                                               gamma,
                                                               m2);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected after compute_delta_modularity_vector: " << cudaGetErrorString(error)
        << "\n";
  }
}

} // namespace nvlouvain

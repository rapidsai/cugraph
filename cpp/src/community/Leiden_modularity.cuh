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

#pragma once
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

namespace {
template<typename IdxT, typename ValT>
__global__ void compute_delta_modularity_vector_constrained(IdxT nnz,
                                                            IdxT num_verts,
                                                            IdxT* csr_off,
                                                            IdxT* csr_ind,
                                                            ValT* edge_weights,
                                                            IdxT* clusters,
                                                            IdxT* constraint,
                                                            ValT* e_c,
                                                            ValT* k_c,
                                                            ValT* m_c,
                                                            ValT* k_vec,
                                                            ValT* delta,
                                                            ValT gamma,
                                                            ValT m2) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < nnz) {
    IdxT startVertex = binsearch_maxle(csr_off, tid, (IdxT) 0, num_verts);
    IdxT endVertex = csr_ind[tid];
    IdxT startCluster = clusters[startVertex];
    IdxT endCluster = clusters[endVertex];
    IdxT startConstraint = constraint[startVertex];
    IdxT endConstraint = constraint[startVertex];

    if (startCluster == endCluster || startConstraint != endConstraint) {
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
        if (neighborCluster == startCluster && neighborId == startVertex) {
          startEdges += edge_weights[i];
          endEdges += edge_weights[i];
        }
      }
      ValT end_e = e_c[endCluster];
      ValT end_k = k_c[endCluster];
      ValT my_k = k_vec[startVertex];
      ValT newEndScore = (end_e + endEdges) / m2
          - gamma * (((end_k + my_k) * (end_k + my_k)) / m2 / m2);
      ValT start_e = e_c[startCluster];
      ValT start_k = k_c[startCluster];
      ValT newStartScore = (start_e - startEdges) / m2
          - gamma * (((start_k - my_k) * (start_k - my_k)) / m2 / m2);
      ValT startM = m_c[startCluster];
      ValT endM = m_c[endCluster];
      ValT finalScore = newEndScore - endM + newStartScore - startM;
      if (finalScore > .0001)
        printf("Vertex: %d, moving from %d to %d finalScore %f, newEndScore %f, oldEndScore %f, newStartScore %f, oldStartScore %f\n",
               startVertex,
               startCluster,
               endCluster,
               finalScore,
               newEndScore,
               endM,
               newStartScore,
               startM);
      delta[tid] = finalScore < .0001 ? 0.0 : finalScore;
    }

    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
void compute_delta_modularity_constrained(IdxT nnz,
                              IdxT num_verts,
                              IdxT* csr_off,
                              IdxT* csr_ind,
                              ValT* edge_weights,
                              IdxT* clusters,
                              IdxT* constraint,
                              ValT* e_c,
                              ValT* k_c,
                              ValT* m_c,
                              ValT* k_vec,
                              ValT* delta,
                              ValT gamma,
                              ValT m2) {
  dim3 grid, block;
  block.x = 512;
  grid.x = min((IdxT) CUDA_MAX_BLOCKS, (nnz / 512 + 1));
  compute_delta_modularity_vector_constrained<<<grid, block, 0, nullptr>>>(nnz,
                                                               num_verts,
                                                               csr_off,
                                                               csr_ind,
                                                               edge_weights,
                                                               clusters,
                                                               constraint,
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
    std::cout << "Cuda error detected after compute_delta_modularity_vector_constrained: "
        << cudaGetErrorString(error)
        << "\n";
  }
}

} // anonymous namespace

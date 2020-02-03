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
template<typename IdxT, typename ValT>
__global__ void findBestMatches(IdxT num_verts,
                                IdxT* csr_off,
                                IdxT* csr_ind,
                                ValT* values,
                                IdxT* clusters,
                                IdxT* bestMatch,
                                ValT* scores) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT start = csr_off[tid];
    IdxT end = csr_off[tid + 1];
    IdxT bestCluster = -1;
    ValT bestScore = 0.0;
    for(IdxT i = start; i < end; i++) {
      IdxT neighborId = csr_ind[i];
      IdxT neighborCluster = clusters[neighborId];
      ValT score = values[i];
      if (score > bestScore){
        bestScore = score;
        bestCluster = neighborCluster;
      }
    }

    IdxT currentCluster = clusters[tid];
    if (currentCluster == bestCluster) {
//      printf("Current cluster %d is the same as best cluster!\n", currentCluster);
      bestScore = 0.0;
    }

    scores[tid] = bestScore;
    bestMatch[tid] = bestCluster;
    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
__global__ void assignMovers_partOne(IdxT num_verts,
                                     IdxT* csr_off,
                                     IdxT* csr_ind,
                                     ValT* scores,
                                     IdxT* movers) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT myMoving = movers[tid];

    if (myMoving == -1) {
      IdxT start = csr_off[tid];
      IdxT end = csr_off[tid + 1];
      ValT myScore = scores[tid];
      bool myScoreBest = true;
      for (IdxT i = start; i < end; i++) {
        IdxT neighborId = csr_ind[i];
        ValT score = scores[neighborId];
        if (score >= myScore && tid < neighborId)
          myScoreBest = false;
      }
      if (myScoreBest && myScore > 0.0) {
        movers[tid] = 1;
      }
      if (myScoreBest && myScore == 0.0) {
        movers[tid] = 0;
      }

//      if (myScoreBest)
//        printf("Thread: %d has best score of: %f\n", tid, myScore);
    }
    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
__global__ void assignMovers_partTwo(IdxT num_verts,
                                     IdxT* csr_off,
                                     IdxT* csr_ind,
                                     ValT* scores,
                                     IdxT* movers,
                                     IdxT* unassigned) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT start = csr_off[tid];
    IdxT end = csr_off[tid + 1];
    IdxT myMoving = movers[tid];
    if (myMoving == -1) {
      bool neighborMoving = false;
      for (IdxT i = start; i < end; i++) {
        IdxT neighborId = csr_ind[i];
        if (movers[neighborId] == 1)
          neighborMoving = true;
      }
      if (neighborMoving) {
        movers[tid] = 0;
        scores[tid] = 0.0;
      }
      else {
        ValT myScore = scores[tid];
        if (myScore == 0.0)
          movers[tid] = 0;
        else {
//          printf("Node %d remains unnassigned with score of: %f\n", tid, scores[tid]);
          atomicAdd(unassigned, 1);
        }
      }
//        *unassigned = 1;
    }

    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT>
__global__ void makeMoves(IdxT num_verts,
                          IdxT* clusters,
                          IdxT* movers,
                          IdxT* bestMatch,
                          IdxT* num_moved) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT amImoving = movers[tid];
    if (amImoving == 1) {
//      IdxT currentCluster = clusters[tid];
//      printf("Node: %d moving from %d to %d\n", tid, currentCluster, bestMatch[tid]);
      clusters[tid] = bestMatch[tid];
      atomicAdd(num_moved, 1);
    }
    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
IdxT makeSwaps(IdxT num_verts,
               IdxT* csr_off,
               IdxT* csr_ind,
               ValT* deltaModularity,
               IdxT* clusters) {
  dim3 grid, block;
  block.x = 512;
  grid.x = min((IdxT)CUDA_MAX_BLOCKS, (num_verts / 512 + 1));

  rmm::device_vector<ValT> scores(num_verts, 0.0);
  rmm::device_vector<IdxT> movers(num_verts, -1);
  rmm::device_vector<IdxT> bestMatch(num_verts, -1);
  rmm::device_vector<IdxT> unassigned(1,0);
  rmm::device_vector<IdxT> swapCount(1,0);
  ValT* scores_ptr = thrust::raw_pointer_cast(scores.data());
  IdxT* movers_ptr = thrust::raw_pointer_cast(movers.data());
  IdxT* bestMatch_ptr = thrust::raw_pointer_cast(bestMatch.data());
  IdxT* unassigned_ptr = thrust::raw_pointer_cast(unassigned.data());
  IdxT* swapCount_ptr = thrust::raw_pointer_cast(swapCount.data());

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected before findBestMatches: " << cudaGetErrorString(error)
        << "\n";
  }

  findBestMatches<<<grid, block, 0, nullptr>>>(num_verts,
                                               csr_off,
                                               csr_ind,
                                               deltaModularity,
                                               clusters,
                                               bestMatch_ptr,
                                               scores_ptr);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "Cuda error detected after findBestMatches: " << cudaGetErrorString(error)
        << "\n";
  }

  IdxT unnassigned = 1;
  while (unnassigned > 0) {
    assignMovers_partOne<<<grid, block, 0, nullptr>>>(num_verts,
                                                      csr_off,
                                                      csr_ind,
                                                      scores_ptr,
                                                      movers_ptr);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << "Cuda error detected after assignMovers_partOne: " << cudaGetErrorString(error) << "\n";
    }
    assignMovers_partTwo<<<grid, block, 0, nullptr>>>(num_verts,
                                                      csr_off,
                                                      csr_ind,
                                                      scores_ptr,
                                                      movers_ptr,
                                                      unassigned_ptr);
    cudaMemcpy(&unnassigned, unassigned_ptr, sizeof(IdxT), cudaMemcpyDefault);
    std::cout << "Assign Movers done: " << unnassigned << " remain unassigned\n";
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << "Cuda error detected after part two: " << cudaGetErrorString(error) << "\n";
    }
    thrust::fill(thrust::cuda::par, unassigned.begin(), unassigned.end(), 0);
  }

  makeMoves<<<grid, block, 0, nullptr>>>(num_verts,
                                         clusters,
                                         movers_ptr,
                                         bestMatch_ptr,
                                         swapCount_ptr);

  IdxT swapsMade = 0;
  cudaMemcpy(&swapsMade, swapCount_ptr, sizeof(IdxT), cudaMemcpyDefault);
  return swapsMade;
}

} // namespace nvlouvain




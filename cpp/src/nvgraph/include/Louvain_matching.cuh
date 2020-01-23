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
  IdxT tid = blockId.x * blockDim.x + threadIdx.x;
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
  IdxT tid = blockId.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT start = csr_off[tid];
    IdxT end = csr_off[tid + 1];
    ValT myScore = scores[tid];
    bool myScoreBest = true;
    for (IdxT i = start; i < end; i++) {
      IdxT neighborId = csr_ind[i];
      ValT score = scores[neighborId];
      if (score > myScore)
        myScoreBest = false;
    }
    if (myScoreBest && myScore > 0.0) {
      movers[tid] = 1;
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
  IdxT tid = blockId.x * blockDim.x + threadIdx.x;
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
      else
        *unassigned = 1;
    }

    tid += gridDim.x * blockDim.x;
  }
}

template<typename IdxT, typename ValT>
__global__ void makeMoves(IdxT num_verts,
                          IdxT* clusters,
                          IdxT* movers,
                          IdxT* bestMatch,
                          IdxT* num_moved) {
  IdxT tid = blockId.x * blockDim.x + threadIdx.x;
  while (tid < num_verts) {
    IdxT amImoving = movers[tid];
    if (amImoving == 1) {
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

  findBestMatches<<<grid, block, 0, nullptr>>>(num_verts,
                                               csr_off,
                                               csr_ind,
                                               deltaModularity,
                                               clusters,
                                               bestMatch.begin(),
                                               scores.begin());

  IdxT unnassigned = 1;
  while (unnassigned > 0) {
    assignMovers_partOne<<<grid, block, 0, nullptr>>>(num_verts,
                                                      csr_off,
                                                      csr_ind,
                                                      scores.begin(),
                                                      movers.begin());
    assignMovers_partTwo<<<grid, block, 0, nullptr>>>(num_verts,
                                                      csr_off,
                                                      csr_ind,
                                                      scores.begin(),
                                                      movers.begin(),
                                                      unassigned.begin());
    cudaMemcpy(&unnassigned, unassigned.begin(), sizeof(IdxT), cudaMemcpydefault);
    unassigned.fill(0);
  }

  makeMoves<<<grid, block, 0, nullptr>>>(num_verts,
                                         clusters,
                                         movers.begin(),
                                         bestMatch.begin(),
                                         swapCount.begin());

  IdxT swapsMade = 0;
  cudaMemcpy(&swapsMade, swapCount.begin(), sizeof(IdxT), cudaMemcpyDefault);
  return swapsMade;
}

} // namespace nvlouvain




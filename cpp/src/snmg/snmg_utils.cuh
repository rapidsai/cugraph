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

// snmg utils
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include <omp.h>
#include <rmm_utils.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace cugraph
{

// Wait for all host threads 
void sync_all() {
  cudaDeviceSynchronize();
  #pragma omp barrier 
}

// enable peer access (all to all)
gdf_error setup_peer_access() {
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();  
  for (int j = 0; j < p; ++j) {
    if (i != j) {
      int canAccessPeer = 0;
      CUDA_TRY(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
      if (canAccessPeer) {
		    cudaDeviceEnablePeerAccess(j, 0);
        cudaError_t status = cudaGetLastError();
        if (!(status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled)) {
        	std::cerr << "Could not Enable Peer Access from" << i << " to " << j << std::endl;
        	return GDF_CUDA_ERROR;
        }
      }
      else {
        std::cerr << "P2P access required from " << i << " to " << j << std::endl;
        return GDF_CUDA_ERROR;
      }
    }
  }
  return GDF_SUCCESS;
}

// Each GPU copies its x_loc to x_glob[offset[device]] on all GPU
template <typename val_t>
gdf_error allgather (size_t* offset, val_t* x_loc, val_t ** x_glob) {
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();  
  size_t n_loc= offset[i+1]-offset[i];

  GDF_TRY(setup_peer_access()); 
  // this causes issues with CUB. TODO :  verify the impact on performance.

  // send the local spmv output (x_loc) to all peers to reconstruct the global vector x_glob 
  // After this call each peer has a full, updated, copy of x_glob
  for (int j = 0; j < p; ++j)
    CUDA_TRY(cudaMemcpyPeer(x_glob[j]+offset[i],j, x_loc,i, n_loc*sizeof(val_t)));
    //CUDA_TRY(cudaMemcpy(x_glob[j]+offset[i], x_loc, n_loc*sizeof(val_t),cudaMemcpyDeviceToDevice));
  
  //Make sure everyone has finished copying before returning
  sync_all();

  return GDF_SUCCESS;
}

/**
 * @tparam val_t The value type
 * @tparam func_t The reduce functor type
 * @param length The length of each array being combined
 * @param x_loc Pointer to the local array
 * @param x_glob Pointer to global array pointers
 * @return Error code
 */
template <typename val_t, typename func_t>
gdf_error treeReduce(size_t length, val_t* x_loc, val_t** x_glob){
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();
  GDF_TRY(setup_peer_access());
  int rank = 1;
  while(rank < p){
    // Copy local data to the receiver's global buffer
    if((i - rank) % (rank * 2) == 0){
      int receiver = i - rank;
      cudaMemcpyPeer(x_glob[receiver], receiver, x_loc, i, length*sizeof(val_t));
    }

    // Sync everything now. This shouldn't be required as cudaMemcpyPeer is supposed to synchronize...
    sync_all();

    // Reduce the data from the receiver's global buffer with its local one
    if(i % (rank * 2) == 0 && i + rank < p){
      rmm_temp_allocator allocator(nullptr);
      func_t op;
      thrust::transform(thrust::cuda::par(allocator).on(nullptr),
                        x_glob[i],
                        x_glob[i] + length,
                        x_loc,
                        x_loc,
                        op);
    }
    rank *= 2;
  }

  // Thread 0 copies it's local result into it's global space
  if (i == 0)
    cudaMemcpy(x_glob[i], x_loc, sizeof(val_t) * length, cudaMemcpyDefault);

  // Sync everything before returning
  sync_all();

  return GDF_SUCCESS;
}

/**
 * @tparam val_t The value type
 * @param length The length of the array being broadcast
 * @param x_loc The local array for each node
 * @param x_glob Pointer to the global array pointers
 * @return Error code
 */
template <typename val_t>
gdf_error treeBroadcast(size_t length, val_t* x_loc, val_t** x_glob){
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();
  GDF_TRY(setup_peer_access());
  int rank = 1;
  while(rank * 2 < p)
    rank *= 2;
  for(; rank >= 1; rank /= 2){
    if(i % (rank * 2) == 0 and i + rank < p){
      int receiver = i + rank;
      cudaMemcpyPeer(x_glob[receiver], receiver, x_glob[i], i, sizeof(val_t) * length);
    }
  }

  // Sync everything before returning
  sync_all();

  return GDF_SUCCESS;
}

void print_mem_usage()
{
  size_t free,total;
  cudaMemGetInfo(&free, &total);  
  std::cout<< std::endl<< "Mem used: "<<total-free<<std::endl;
}

} //namespace cugraph

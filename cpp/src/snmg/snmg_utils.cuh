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
		    cudaError_t status = cudaDeviceEnablePeerAccess(j, 0);
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
  // send the local spmv output (x_loc) to all peers to reconstruct the global vector x_glob 
  // After this call each peer has a full, updated, copy of x_glob
  for (int j = 0; j < p; ++j)
    CUDA_TRY(cudaMemcpy(x_glob[j]+offset[i], x_loc, n_loc*sizeof(val_t),cudaMemcpyDeviceToDevice));
  
  //Make sure everyone has finished copying before returning
  sync_all();

  return GDF_SUCCESS;
}

} //namespace cugraph

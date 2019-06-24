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
#include "rmm_utils.h"

namespace cugraph
{

// basic info about the snmg env setup
class SNMGinfo 
{ 
  private:
    int i, p, n_sm;
  
  public: 
    SNMGinfo() { 
      int tmp_p, tmp_i;
      //get info from cuda
      cudaGetDeviceCount(&tmp_p);
      cudaGetDevice(&tmp_i);

      //get info from omp 
      i = omp_get_thread_num();
      p = omp_get_num_threads();

      // check that thread_num and num_threads are compatible with the device ID and the number of device 
      if (tmp_i != i) {
        std::cerr << "Thread ID and GPU ID do not match" << std::endl;
      }
      if (p > tmp_p) {
        std::cerr << "More threads than GPUs" << std::endl;
      }
      // number of SM, usefull for kernels paramters
      cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, i);
      cudaCheckError();
    } 
    ~SNMGinfo() { }

    int get_thread_num() {
      return i; 
    }
    int get_num_threads() {
      return p; 
    }
    int get_num_sm() {
      return n_sm; 
    } 
    // enable peer access (all to all)
    void setup_peer_access() {
      for (int j = 0; j < p; ++j) {
        if (i != j) {
          int canAccessPeer = 0;
          cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
          cudaCheckError();
          if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(j, 0);
            cudaError_t status = cudaGetLastError();
            if (!(status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled)) {
              std::cerr << "Could not Enable Peer Access from" << i << " to " << j << std::endl;
            }
          }
          else {
            std::cerr << "P2P access required from " << i << " to " << j << std::endl;
          }
        }
      }
    }
};

// Wait for all host threads 
void sync_all();

// Each GPU copies its x_loc to x_glob[offset[device]] on all GPU
template <typename val_t>
void allgather (SNMGinfo & env, size_t* offset, val_t* x_loc, val_t ** x_glob) {
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();  
  size_t n_loc= offset[i+1]-offset[i];

  env.setup_peer_access(); 
  // this causes issues with CUB. TODO :  verify the impact on performance.

  // send the local spmv output (x_loc) to all peers to reconstruct the global vector x_glob 
  // After this call each peer has a full, updated, copy of x_glob
  for (int j = 0; j < p; ++j) {
    cudaMemcpyPeer(x_glob[j]+offset[i],j, x_loc,i, n_loc*sizeof(val_t));
    cudaCheckError();
  }
  
  //Make sure everyone has finished copying before returning
  sync_all();

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
gdf_error treeReduce(SNMGinfo& env, size_t length, val_t* x_loc, val_t** x_glob){
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();
  env.setup_peer_access();
  int rank = 1;
  while(rank < p){
    // Copy local data to the receiver's global buffer
    if((i - rank) % (rank * 2) == 0){
      int receiver = i - rank;
      cudaMemcpyPeer(x_glob[receiver], receiver, x_loc, i, length*sizeof(val_t));
      cudaCheckError();
    }

    // Sync everything now. This shouldn't be required as cudaMemcpyPeer is supposed to synchronize...
    sync_all();

    // Reduce the data from the receiver's global buffer with its local one
    if(i % (rank * 2) == 0 && i + rank < p){
      func_t op;
      thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                        x_glob[i],
                        x_glob[i] + length,
                        x_loc,
                        x_loc,
                        op);
      cudaCheckError();
    }
    sync_all();
    rank *= 2;
  }

  // Thread 0 copies it's local result into it's global space
  if (i == 0) {
    cudaMemcpy(x_glob[i], x_loc, sizeof(val_t) * length, cudaMemcpyDefault);
    cudaCheckError();
  }

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
gdf_error treeBroadcast(SNMGinfo& env, size_t length, val_t* x_loc, val_t** x_glob){
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();
  env.setup_peer_access();
  int rank = 1;
  while(rank * 2 < p)
    rank *= 2;
  for(; rank >= 1; rank /= 2){
    if(i % (rank * 2) == 0 and i + rank < p){
      int receiver = i + rank;
      cudaMemcpyPeer(x_glob[receiver], receiver, x_glob[i], i, sizeof(val_t) * length);
      cudaCheckError();
    }
    sync_all();
  }

  // Sync everything before returning
  sync_all();

  return GDF_SUCCESS;
}

void print_mem_usage();

} //namespace cugraph

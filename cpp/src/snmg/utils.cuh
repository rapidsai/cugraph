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
inline void sync_all() {
  cudaDeviceSynchronize();
  #pragma omp barrier 
  cudaCheckError();
}

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
      sync_all();
    }
};

// Each GPU copies its x_loc to x_glob[offset[device]] on all GPU
template <typename val_t>
void allgather (SNMGinfo & env, size_t* offset, val_t* x_loc, val_t ** x_glob) {
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();  
  size_t n_loc= offset[i+1]-offset[i];

  env.setup_peer_access(); 
  // send the local spmv output (x_loc) to all peers to reconstruct the global vector x_glob 
  // After this call each peer has a full, updated, copy of x_glob
  for (int j = 0; j < p; ++j) {
    cudaMemcpyPeer(x_glob[j]+offset[i],j, x_loc,i, n_loc*sizeof(val_t));
    cudaCheckError();
  }
  
  //Make sure everyone has finished copying before returning
  sync_all();
}

inline void print_mem_usage()
{
  size_t free,total;
  cudaMemGetInfo(&free, &total);  
  std::cout<< std::endl<< "Mem used: "<<total-free<<std::endl;
}

} //namespace cugraph

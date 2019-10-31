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

#include <omp.h>
#include <iostream>
#include <snmg/utils.cuh>

namespace cugraph { 
namespace snmg {

static bool PeerAccessAlreadyEnabled = false; 

// basic info about the snmg env setup
SNMGinfo::SNMGinfo() { 
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
  CUDA_CHECK_LAST();
 } 
 SNMGinfo::~SNMGinfo() { }

 int SNMGinfo::get_thread_num() {
   return i; 
 }
 int SNMGinfo::get_num_threads() {
   return p; 
 }
 int SNMGinfo::get_num_sm() {
   return n_sm; 
 } 
 // enable peer access (all to all)
 void SNMGinfo::setup_peer_access() {
   if (PeerAccessAlreadyEnabled)
     return;
   for (int j = 0; j < p; ++j) {
     if (i != j) {
          int canAccessPeer = 0;
          cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
          CUDA_CHECK_LAST();
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
  PeerAccessAlreadyEnabled = true;
}
void sync_all() {
  cudaDeviceSynchronize();
  #pragma omp barrier
}

void print_mem_usage() {
  size_t free,total;
  cudaMemGetInfo(&free, &total);
  std::cout<< std::endl<< "Mem used: "<<total-free<<std::endl;
}

} } //namespace

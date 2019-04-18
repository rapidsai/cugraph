
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
#pragma once

#include <string>
#include "test_opt_utils.h"
#include "graph_utils.cuh"
#include "louvain.cuh"
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "util.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


template<typename IdxIter, typename ValIter, typename ValType>
__global__ void 
kernel_delta_modularity(const int n_vertex, IdxIter csr_ptr_iter, IdxIter csr_ind_iter, ValIter csr_val_iter, IdxIter cluster, ValType* score){

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if( i<n_vertex && c < n_vertex ){
    nvlouvain::delta_modularity_block( n_vertex, csr_ptr_iter, csr_ind_iter, csr_val_iter, cluster, i, c, &score[i*n_vertex +c] );
    //printf("i: %d c: %d delta: %f\n", i, c, score[i*n_vertex +c] );
  }

}


void delta_modularity_test(thrust::device_vector<int> &csr_ptr_d,
                     thrust::device_vector<int> &csr_ind_d,
                     thrust::device_vector<T> &csr_val_d,
                     const int size){

  HighResClock hr_clock;
  double timed;
  
  dim3 block_size((size + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, (size + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, 1);
  dim3 grid_size(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1); 

 
  thrust::device_vector<int> cluster_d(size);
  thrust::sequence(cluster_d.begin(), cluster_d.end());  
  std::cout<<"cluster: ";
  nvlouvain::display_vec(cluster_d);

  thrust::device_vector<T> score_d(size*size);
  T* score_d_raw_ptr = thrust::raw_pointer_cast(score_d.data());


  hr_clock.start();

  kernel_delta_modularity<<<block_size, grid_size>>>(size, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), cluster_d.begin(), score_d_raw_ptr);

 
  CUDA_CALL(cudaDeviceSynchronize());

  hr_clock.stop(&timed);
  double mod_time(timed);
  std::cout<<"delta modularity: "<<score_d[0]<<" runtime: "<<mod_time<<std::endl;


  
}





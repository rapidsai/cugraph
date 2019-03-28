
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

template< typename IdxIter, typename ValIter, typename ValType >
__global__ void kernal_k_in_test(const int size, IdxIter csr_ptr_iter, IdxIter csr_ind_iter, ValIter csr_val_iter, IdxIter cluster_iter, int i, ValType* result){
/*
  
  //printf("successfully launch kernal\n");

  int idx_x = blockDim.x*blockIdx.x + threadIdx.x;
  int idx_y = blockDim.y*blockIdx.y + threadIdx.y;

  if(idx_x < size && idx_y < size ){
    
    int c = *( cluster_iter + idx_y);  
    //printf(" ** %d %d\n", idx_x, idx_y); 
    //printf("start compute k with iter passing. (%d, %d, %d) idx = %d %f\n", blockDim.x, blockIdx.x, threadIdx.x, idx, result[idx]);
    nvlouvain::compute_k_i_in(size, csr_ptr_iter, csr_ind_iter, csr_val_iter, cluster_iter, c, idx_x, &result[idx_x *size + idx_y ]);
                       // n_vertex, csr_ptr_iter, csr_idx_iter, csr_val_iter, cluster_iter,      c,   i, result
    printf("k_%d_in_c%d = %f\n", idx_x, idx_y ,result[idx_x *size + idx_y]);
    
  }
*/
/*
  if(idx == 0){
    nvlouvain::display_vec(csr_ptr_iter, size);  
    nvlouvain::display_vec(csr_ind_iter, csr_ptr_iter[size]);
    nvlouvain::display_vec(csr_val_iter, csr_ptr_iter[size]);

  }
*/
  return;

}


void k_i_in_compute_test( thrust::device_vector<int> &csr_ptr_d,
                          thrust::device_vector<int> &csr_ind_d,
                          thrust::device_vector<T> &csr_val_d,
                          int size){

  HighResClock hr_clock;
  double timed;

  
  dim3 block_size((size + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, (size + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, 1);
  dim3 grid_size(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);

  std::cout<< csr_ptr_d.size()<<" "<<csr_val_d.size()<<" size:"<< size <<std::endl;
  thrust::device_vector<T> result_d(size * size);
  thrust::device_vector<int> cluster_d(size);

  T* result_ptr = thrust::raw_pointer_cast(result_d.data());


  hr_clock.start();
  int i = 0; 
  std::cout<<"successfully declair device vector.\n";
  kernal_k_in_test<<<block_size, grid_size>>>(size, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), cluster_d.begin(), i, result_ptr);
  CUDA_CALL(cudaDeviceSynchronize());

  hr_clock.stop(&timed);
  double iter_time(timed);
  nvlouvain::display_vec(result_d);

  std::cout<<"k_i_in runtime: "<<iter_time<<"\n";
 std::cout<<"============== complete k_i_in computation test =============\n";
 
}

/*
void k_i_in_compute_for_each_with_functor(thrust::device_vector<int> &csr_ptr_d,
                                          thrust::device_vector<int> &csr_ind_d,
                                          thrust::device_vector<T> &csr_val_d,
                                          int size){
  for_each_n() 
}*/


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

/*
template< typename IdxType, typename ValType >
__global__ void kernal_test(const int size, IdxType* csr_ptr, ValType* csr_val, int i, ValType* result){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < size){
    nvlouvain::compute_k(size, csr_ptr, csr_val, idx, &result[idx]);
    //printf("k%d = %f\n", idx ,result[idx]);
    
  }
  return;

}

template< typename IdxIter, typename ValIter, typename ValType >
__global__ void kernal_test_iter(const int size, IdxIter csr_ptr_iter, ValIter csr_val_iter, int i, ValType* result){

  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < size){

    //printf("start compute k with iter passing. (%d, %d, %d) idx = %d %f\n", blockDim.x, blockIdx.x, threadIdx.x, idx, result[idx]);
    nvlouvain::compute_k(size, csr_ptr_iter, csr_val_iter, idx, &result[idx]);

    //printf("k%d = %f\n", idx ,result[idx]);
    
  }
  return;

}


template< typename IdxIter, typename ValIter, typename DevPtr >
__global__ void kernal_test_dev_ptr(const int size, IdxIter csr_ptr_iter, ValIter csr_val_iter, int i, DevPtr result){

  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < size){
    //printf("start compute k with iter passing. (%d, %d, %d) idx = %d %f\n", blockDim.x, blockIdx.x, threadIdx.x, idx, result[idx]);
    nvlouvain::compute_k(size, csr_ptr_iter, csr_val_iter, idx, &result[idx]);
    //printf("k%d = %f\n", idx ,result[idx]);
  }
  return;

}



void k_compute_test( thrust::device_vector<int> &csr_ptr_d,
                     thrust::device_vector<int> &csr_ind_d,
                     thrust::device_vector<T> &csr_val_d,
                     int size){

  HighResClock hr_clock;
  double timed;

  
  dim3 block_size((size + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1);

  
  std::cout<< csr_ptr_d.size()<<" "<<csr_val_d.size()<<" size:"<< size <<std::endl;

  int* csr_ptr_d_raw_ptr = thrust::raw_pointer_cast(csr_ptr_d.data()); 
  T* csr_val_d_raw_ptr = thrust::raw_pointer_cast(csr_val_d.data());

  thrust::device_vector<T> k_d(size);
  T* k_d_raw_cast_ptr = thrust::raw_pointer_cast(k_d.data());  

  hr_clock.start();
  kernal_test<<<block_size,grid_size>>>(size , csr_ptr_d_raw_ptr, csr_val_d_raw_ptr, 0, k_d_raw_cast_ptr);
  CUDA_CALL(cudaDeviceSynchronize());
//  nvlouvain::display_vec(k_d);
  hr_clock.stop(&timed); 
  double raw_ptr_time(timed);  



  thrust::device_vector<T> k_iter_d(size);
  T* k_iter_d_raw_ptr = thrust::raw_pointer_cast(k_iter_d.data());
  hr_clock.start();
  kernal_test_iter<<<block_size, grid_size>>>(size, csr_ptr_d.begin(), csr_val_d.begin(), 0, k_iter_d_raw_ptr);
  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed);
  double iter_time(timed);
//  nvlouvain::display_vec(k_iter_d);


  thrust::device_vector<T> k_d_ptr_d(size);
  hr_clock.start();
  kernal_test_dev_ptr<<<block_size, grid_size>>>(size, csr_ptr_d.begin(), csr_val_d.begin(), 0, k_d_ptr_d.data());
  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed);
  double dev_ptr_time(timed);
//  nvlouvain::display_vec(k_d_ptr_d);




  std::cout<<"raw_ptr_runtime: "<<raw_ptr_time<<"\niter_time: "<<iter_time<<"\ndev_ptr_time: "<<dev_ptr_time<<std::endl;
  std::cout<<"============== complete k computation test =============\n";
  
}
*/

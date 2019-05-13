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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>




template<typename iter, typename ptr >
__global__ void test_sum(iter begin, iter end, ptr sum){

  thrust::plus<T> op;
  *sum = thrust::reduce(thrust::cuda::par, begin, end, 0.0, op);
  
}

__global__ void test_sum_cast(T* vec, size_t size, T* sum){

  thrust::plus<T> op;
  *sum = thrust::reduce(thrust::cuda::par, vec, vec+size, 0.0, op);
  
}


void thrust_passing_arg_test( thrust::host_vector<int> &csr_ptr_h, 
                              thrust::host_vector<int> &csr_ind_h,
                              thrust::host_vector<T> &csr_val_h,
                              thrust::device_vector<int> &csr_ptr_d, 
                              thrust::device_vector<int> &csr_ind_d, 
                              thrust::device_vector<T> &csr_val_d){

  HighResClock hr_clock;
  double timed;
  
  thrust::plus<T> binary_op;
  hr_clock.start();
  T sum_h = thrust::reduce(csr_val_h.begin(), csr_val_h.end(), 0.0, binary_op); 
  hr_clock.stop(&timed);
  double cpu_time(timed);
 
  
  
  thrust::copy(csr_val_d.begin(), csr_val_d.end(), std::ostream_iterator<float>(std::cout, " "));
  std::cout<<std::endl;


  dim3 block_size(1, 1, 1);
  dim3 grid_size(1, 1, 1);
  

  hr_clock.start();
  T sum_r = thrust::reduce(csr_val_d.begin(), csr_val_d.end(), 0.0, binary_op);

  hr_clock.stop(&timed);
  double r_time(timed);



  hr_clock.start();
  thrust::device_vector<T> sum_d(1, 0.0);
  test_sum<<<block_size,grid_size>>>( csr_val_d.begin(),csr_val_d.end(), sum_d.data());
  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed);
  double cuda_time(timed);

  
  hr_clock.start();
  cudaStream_t s;
  thrust::device_vector<T> sum_a(1, 0.0);
  cudaStreamCreate(&s);
  test_sum<<<1,1,0,s>>>(csr_val_d.begin(),csr_val_d.end(), sum_a.data());
  cudaStreamSynchronize(s);
  hr_clock.stop(&timed);
  double asyn_time(timed);



  hr_clock.start();
  T* csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  double* raw_sum;  
  double sum_cast;
  cudaMalloc((void **) &raw_sum, sizeof(double));
  test_sum_cast<<<block_size,grid_size>>>( csr_val_ptr, csr_val_d.size(), raw_sum);
  cudaMemcpy(&sum_cast, raw_sum, sizeof(double),cudaMemcpyDeviceToHost);
  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed);
  double cast_time(timed);
  cudaFree(raw_sum);



  
  std::cout<<"cpu    sum of val: "<< sum_h <<" runtime: "<<cpu_time<<std::endl;
  std::cout<<"device sum of val: "<< sum_r <<" runtime: "<<r_time<<std::endl;
  std::cout<<"kernel sum of val: "<< sum_d[0] <<" runtime: "<<cuda_time<<std::endl;
  std::cout<<"async  sum of val: "<< sum_a[0] <<" runtime: "<<asyn_time<<std::endl;
  std::cout<<"cast:  sum of val: "<< sum_cast <<" runtime: "<<cast_time<<std::endl;

} 

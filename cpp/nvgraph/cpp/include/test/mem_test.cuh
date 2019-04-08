
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
#include <thrust/memory.h>



template<typename IdxType=int, typename ValType=double>
__global__ void
kernel_local_mem(const int n_vertex ){

  thrust::device_system_tag device_sys;
  thrust::pointer<ValType,thrust::device_system_tag> temp_i = thrust::malloc<ValType>(device_sys, n_vertex); // for weight on i and for sum_k
  thrust::pointer<IdxType,thrust::device_system_tag> temp_idx = thrust::malloc<IdxType>(device_sys, n_vertex); // for weight on i and for sum_k

  

  *temp_i = 10.0;
  *(temp_i + n_vertex-1) = 100.5;
  
  thrust::return_temporary_buffer(device_sys, temp_idx);
  thrust::return_temporary_buffer(device_sys, temp_i);
}

template<typename IdxType=int, typename ValType=double>
__global__ void
kernel_local_mem_new(const int n_vertex ){

  ValType * temp_i = new ValType[n_vertex];
  IdxType * temp_idx = new IdxType[n_vertex];
 

  *temp_i = 10.0;
  *(temp_i + n_vertex-1) = 100.5;
  thrust::sequence(thrust::cuda::par, temp_idx, temp_idx + n_vertex);
  printf("%d %d %d ... %d\n",*temp_idx, *(temp_idx+1), *(temp_idx+2), *(temp_idx + n_vertex - 1) );

  delete [] temp_i;    
  delete [] temp_idx;
}




void mem_allocate_test(const int size){
 
 
  HighResClock hr_clock;
  double timed;

 
  dim3 block_size((size + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1);
  hr_clock.start();

  kernel_local_mem<<<block_size,grid_size>>>(30000);

  kernel_local_mem_new<<<block_size,grid_size>>>(30000);


  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed); 
  double raw_ptr_time(timed);  

  std::cout<<"allocate_mem_runtime: "<<raw_ptr_time<<std::endl;


 
   
}

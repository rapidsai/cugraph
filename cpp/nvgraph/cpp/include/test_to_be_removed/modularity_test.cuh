
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

#include <fstream>
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

  
void modularity_test_no_matrix(thrust::device_vector<int> &csr_ptr_d,
                     thrust::device_vector<int> &csr_ind_d,
                     thrust::device_vector<T> &csr_val_d,
                     const int size, 
                     const bool weighted){
  

  HighResClock hr_clock;
  double timed;

 
 
  dim3 block_size((size + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1); 


  std::cout<<"n_vertex: "<<size<<std::endl;
  std::vector<int> cluster;
  
 
  thrust::device_vector<int> cluster_d(size);
//  thrust::sequence(cluster_d.begin(), cluster_d.end());  
  
//  std::cout<<"cluster: ";
  //nvlouvain::display_vec(cluster_d);

  thrust::device_vector<T> score(1);
  thrust::device_vector<T> k_vec(size);
  thrust::device_vector<T> Q_arr(size);
  thrust::device_vector<T> temp_i(csr_ptr_d[size]);
  thrust::device_vector<int> cluster_inv_ptr(size+1);
  thrust::device_vector<int> cluster_inv_ind(size);
  thrust::sequence(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.end());
  thrust::sequence(thrust::cuda::par, cluster_inv_ind.begin(), cluster_inv_ind.end());
  thrust::fill(thrust::device, temp_i.begin(), temp_i.end(), 0.0);

//  nvlouvain::display_vec(temp_i);

  T* score_ptr = thrust::raw_pointer_cast(score.data());
  T* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  T* Q_arr_ptr = thrust::raw_pointer_cast(Q_arr.data());
  T* temp_i_ptr = thrust::raw_pointer_cast(temp_i.data());
  int* csr_ptr_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  int* csr_ind_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
  T* csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  int* cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
  int* cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());
  int* cluster_ptr = thrust::raw_pointer_cast(cluster_d.data());

 
  
  hr_clock.start();

  T m2 = thrust::reduce(thrust::cuda::par, csr_val_d.begin(), csr_val_d.end());
  nvlouvain::generate_cluster_inv(size, size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);

  double Q = nvlouvain::modularity(size, csr_ptr_d[size], size, m2,
                                   csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
                                   cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                                   weighted, k_vec_ptr, Q_arr_ptr, temp_i_ptr);
/* 
  nvlouvain::kernel_modularity_no_matrix<<<block_size, grid_size >>>(size, size, m2,
                                                                     csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), 
                                                                     cluster_d.begin(), cluster_inv_ptr.begin(), cluster_inv_ind.begin(),
                                                                     weighted, k_vec_ptr, Q_arr_ptr, temp_i_ptr, score_ptr);

     
  CUDA_CALL(cudaDeviceSynchronize());
  double Q = score[0];
*/
  hr_clock.stop(&timed);
  double mod_time(timed);
  printf("modularity(w/o block): %.10e  runtime: ",Q);
  std::cout<<mod_time<<std::endl;
 

  /* 
  for(auto const & it:Q_arr) {
    std::cout<<it<<" ,";
  }
  std::cout<<std::endl;
*/
} 
void modularity_test_no_matrix_block(thrust::device_vector<int> &csr_ptr_d,
                     thrust::device_vector<int> &csr_ind_d,
                     thrust::device_vector<T> &csr_val_d,
                     const int size, 
                     const bool weighted){
  
  HighResClock hr_clock;
  double timed;
   
  dim3 block_size((size + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1); 

  std::cout<<"n_vertex: "<<size<<std::endl;
 
  thrust::device_vector<int> cluster_d(size);
  thrust::sequence(cluster_d.begin(), cluster_d.end());  
  //std::cout<<"cluster: ";
  //nvlouvain::display_vec(cluster_d);

  thrust::device_vector<T> score(1);
  thrust::device_vector<T> k_vec(size);
  thrust::device_vector<T> Q_arr(size);

  T* score_ptr = thrust::raw_pointer_cast(score.data());
  T* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  T* Q_arr_ptr = thrust::raw_pointer_cast(Q_arr.data());

  int n_edges = csr_ptr_d[size];
  T m2 = thrust::reduce(thrust::cuda::par, csr_val_d.begin(), csr_val_d.end()+ n_edges);

  hr_clock.start();

  
  nvlouvain::kernel_modularity_no_matrix_block<<<block_size, grid_size>>>(size, m2, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), 
                                                                          cluster_d.begin(), 
                                                                          weighted, k_vec_ptr, Q_arr_ptr);
 
  CUDA_CALL(cudaDeviceSynchronize());

  hr_clock.stop(&timed);
  double mod_time(timed);
  double Q = thrust::reduce(thrust::cuda::par, Q_arr_ptr, Q_arr_ptr + size, (0.0));     

  printf("modularity(w/  block): %.10e  runtime: ",Q);
  std::cout<<mod_time<<std::endl;
/*
 
  for(auto const & it:Q_arr) {
    std::cout<<it<<" ,";
  }
  std::cout<<std::endl;
*/

}
 
  /* 
void modularity_test_no_matrix(std::string file_name){
  

  HighResClock hr_clock;
  double timed;
  std::ifstream inf(file_name);

 
  thrust::device_vector<int> csr_ptr_d; 
  thrust::device_vector<int> csr_ind_d,
  thrust::device_vector<T> csr_val_d;
  const int size;
  bool weighted = truel
  dim3 block_size((size + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1); 


  std::cout<<"n_vertex: "<<size<<std::endl;
  std::vector<int> cluster;
  
 
  thrust::device_vector<int> cluster_d(size);
//  thrust::sequence(cluster_d.begin(), cluster_d.end());  
  
//  std::cout<<"cluster: ";
  //nvlouvain::display_vec(cluster_d);

  thrust::device_vector<T> score(1);
  thrust::device_vector<T> k_vec(size);
  thrust::device_vector<T> Q_arr(size);
  thrust::device_vector<T> temp_i(csr_ptr_d[size]);
  thrust::device_vector<int> cluster_inv_ptr(size+1);
  thrust::device_vector<int> cluster_inv_ind(size);
  thrust::sequence(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.end());
  thrust::sequence(thrust::cuda::par, cluster_inv_ind.begin(), cluster_inv_ind.end());
  thrust::fill(thrust::device, temp_i.begin(), temp_i.end(), 0.0);

//  nvlouvain::display_vec(temp_i);

  T* score_ptr = thrust::raw_pointer_cast(score.data());
  T* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  T* Q_arr_ptr = thrust::raw_pointer_cast(Q_arr.data());
  T* temp_i_ptr = thrust::raw_pointer_cast(temp_i.data());
  
  hr_clock.start();

  T m2 = thrust::reduce(thrust::cuda::par, csr_val_d.begin(), csr_val_d.end());
  nvlouvain::generate_cluster_inv(size, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);

  double Q = nvlouvain::modularity(size, size, m2,
                                   csr_ptr_d, csr_ind_d, csr_val_d,
                                   cluster_d, cluster_inv_ptr, cluster_inv_ind,
                                   weighted, k_vec_ptr, Q_arr_ptr, temp_i_ptr);
  nvlouvain::kernel_modularity_no_matrix<<<block_size, grid_size >>>(size, size, m2,
                                                                     csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), 
                                                                     cluster_d.begin(), cluster_inv_ptr.begin(), cluster_inv_ind.begin(),
                                                                     weighted, k_vec_ptr, Q_arr_ptr, temp_i_ptr, score_ptr);

     
  CUDA_CALL(cudaDeviceSynchronize());
  double Q = score[0];
  hr_clock.stop(&timed);
  double mod_time(timed);
  printf("modularity(w/o block): %.10e  runtime: ",Q);
  std::cout<<mod_time<<std::endl;
 
}
*/

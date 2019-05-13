

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
#include <vector>
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



void cluster_inv_test(){
  std::vector<int> cluster = {0, 1, 1, 2, 1, 0, 2, 2, 3, 4, 5, 6, 4, 6, 5, 3};
  int n_vertex = 16;
  int c_size = 7;
  thrust::device_vector<int> cluster_d(cluster.begin(), cluster.end());
  thrust::device_vector<int> cluster_inv_ptr(c_size + 1);
  thrust::device_vector<int> cluster_inv_ind(n_vertex);
  int* cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
  int* cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());
  thrust::device_vector<int> seq_idx(n_vertex);
  thrust::sequence(seq_idx.begin(), seq_idx.end());
  int* seq_idx_ptr =  thrust::raw_pointer_cast(seq_idx.data());

  dim3 block_size((n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1); 


  nvlouvain::generate_cluster_inv(n_vertex, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);
  #ifdef VERBOSE
    nvlouvain::display_vec(cluster_inv_ptr);
    nvlouvain::display_vec(cluster_inv_ind);
  #endif
//  nvlouvain::display_vec_size(cluster_inv_ind_ptr, n_vertex);

}


void cluster_sum_test(thrust::device_vector<int> &csr_ptr_d,
                     thrust::device_vector<int> &csr_ind_d,
                     thrust::device_vector<T> &csr_val_d,
                     const int n_vertex,
                     bool weighted){

  HighResClock hr_clock;
  double timed, diff_time;
  std::vector<int> cluster(n_vertex);
  int c_size;

  if(n_vertex == 16){ 
    cluster = {0, 1, 1, 2, 1, 0, 2, 2, 3, 4, 5, 6, 4, 6, 5, 3};
    c_size = 7;
  }
  else{
    for(int i = 0 ; i <n_vertex ; ++i){
      cluster[i]=i;
    }
    c_size = n_vertex;
  }
  thrust::device_vector<int> cluster_d(cluster.begin(), cluster.end());
  thrust::device_vector<int> cluster_inv_ptr(c_size+1);
  thrust::device_vector<int> cluster_inv_ind(n_vertex);
  int* cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
  int* cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());
  thrust::device_vector<int> seq_idx(n_vertex);
  thrust::sequence(seq_idx.begin(), seq_idx.end());
  int* seq_idx_ptr =  thrust::raw_pointer_cast(seq_idx.data());

  dim3 block_size((n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size(BLOCK_SIZE_1D, 1, 1); 



  thrust::device_vector<T> score(1);
  thrust::device_vector<T> k_vec(n_vertex);
  thrust::device_vector<T> Q_arr(n_vertex);
  thrust::device_vector<T> delta_Q_arr(csr_ptr_d[n_vertex]);
  thrust::device_vector<T> cluster_sum_vec(c_size);
  

  T* score_ptr = thrust::raw_pointer_cast(score.data());
  T* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  T* Q_arr_ptr = thrust::raw_pointer_cast(Q_arr.data());
  T* cluster_sum_vec_ptr = thrust::raw_pointer_cast(cluster_sum_vec.data());
  T* delta_Q_arr_ptr =  thrust::raw_pointer_cast(delta_Q_arr.data());
  int* csr_ptr_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  int* csr_ind_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
  T* csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  int* cluster_ptr = thrust::raw_pointer_cast(cluster_d.data());



  hr_clock.start();
  nvlouvain::generate_cluster_inv(n_vertex, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);
  hr_clock.stop(&timed);
  diff_time = timed;

  weighted = true;

  #ifdef VERBOSE
    printf("cluster inv: \n");
    nvlouvain::display_vec(cluster_inv_ptr);
    nvlouvain::display_vec(cluster_inv_ind);
  #endif
  std::cout<<"cluster inv rumtime: "<<diff_time<<" us\n";
  T m2 = thrust::reduce(thrust::cuda::par, csr_val_d.begin(), csr_val_d.end());

  hr_clock.start();
  double Q = nvlouvain::modularity(n_vertex, csr_ptr_d[n_vertex],c_size, m2, 
                        csr_ptr_ptr, csr_ind_ptr, csr_val_ptr, 
                        cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
                        weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr);

/*
  nvlouvain::kernel_modularity_no_matrix<<<block_size, grid_size >>>(n_vertex, c_size, m2,
                                                                     csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), 
                                                                     cluster_d.begin(), cluster_inv_ptr.begin(), cluster_inv_ind.begin(),
                                                                     weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr,score_ptr);
 
  CUDA_CALL(cudaDeviceSynchronize());

  double Q = score[0];
*/
  hr_clock.stop(&timed);
  diff_time = timed;

  #ifdef VERBOSE
    printf("Q_arr: \n");
    nvlouvain::display_vec(Q_arr);
    printf("k_vec: \n");
    nvlouvain::display_vec(k_vec);
  #endif
  printf("modularity(w/o block): %.10e  runtime: ",Q);
  std::cout<<diff_time<<std::endl;

  //====================  

  int side = (n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D;
  dim3 block_size_2d(side,side,1);
  dim3 grid_size_2d(BLOCK_SIZE_1D, BLOCK_SIZE_1D, 1);
 
 
  hr_clock.start();
  nvlouvain::build_delta_modularity_vec<<<block_size_2d, grid_size_2d>>>(n_vertex, 
                                                                         csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), 
                                                                         cluster_d.begin(), delta_Q_arr_ptr);
  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed);
  diff_time = timed;
  #ifdef VERBOSE
    nvlouvain::display_vec(Q_arr);
  #endif
  std::cout<<"delta (w block) rumtime: "<<diff_time<<" us\n";
  





  //====================

/*
  hr_clock.start();
  nvlouvain::kernel_compute_cluster_sum<<<block_size, grid_size>>>( n_vertex, c_size, 
                                                                      cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
                                                                      k_vec_ptr, cluster_sum_vec_ptr);
  CUDA_CALL(cudaDeviceSynchronize());
  #ifdef VERBOSE
    nvlouvain::display_vec(cluster_sum_vec);
  #endif
  nvlouvain::build_delta_modularity_vec<<<block_size_2d, grid_size_2d>>>(n_vertex, c_size, m2 
                                                                         csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), 
                                                                         cluster_d.begin(),
                                                                         cluster_sum_vec_ptr, 
                                                                         k_vec_ptr, delta_Q_arr_ptr);
  CUDA_CALL(cudaDeviceSynchronize());
  hr_clock.stop(&timed);
  diff_time = timed;
  #ifdef VERBOSE
    nvlouvain::display_vec(Q_arr);
  #endif

  std::cout<<"delta (wo block)rumtime: "<<diff_time<<" us\n";
*/
 
} 

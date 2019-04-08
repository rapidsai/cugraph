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

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/transform.h>

#include "util.cuh"
#include "graph_utils.cuh"
#include "functor.cuh"
//#include "block_modulariy.cuh"


namespace nvlouvain{
/*************************************************************
*
*  compute k vector from [ k0, k1, ..., kn ]
*
*  - input :
*     n_vertex
*     csr_ptr's iterator 
*     csr_val's iterator
*
*  - output:
*     results: k_vec : k vectors
*
***************************************************************/
template<typename ValType, typename IdxType>
__device__ void compute_k_vec(const int n_vertex, IdxType* csr_ptr_ptr, ValType* csr_val_ptr, bool weighted, ValType* k_vec){

  int tid = blockDim.x*blockIdx.x + threadIdx.x;

  if( (tid < n_vertex) ){

    int start_idx = *(csr_ptr_ptr + tid);
    int end_idx = *(csr_ptr_ptr + tid + 1);

#ifdef DEBUG
    if( end_idx > (*(csr_ptr_ptr + n_vertex)) ){
      printf("Error computing ki iter but end_idx >= n_vertex %d >= %d\n");
      *(k_vec + tid) = 0.0;
    }
#endif

    if(!weighted){
      *(k_vec + tid) = (ValType)end_idx - start_idx;
    }
    else{
      ValType sum = 0.0;    
#pragma unroll 
      for(int i = 0 ; i < end_idx - start_idx; ++ i){
        sum += *(csr_val_ptr + start_idx + i);
      }    
      *(k_vec + tid) = sum;
    }
  }
  return; 
}

template<typename IdxType, typename ValType> 
__device__ void
modularity_i( const int n_vertex, 
              const int n_clusters,
              IdxType* csr_ptr_ptr, 
              IdxType* csr_ind_ptr, 
              ValType* csr_val_ptr, 
              IdxType* cluster_ptr, 
              IdxType* cluster_inv_ptr_ptr,
              IdxType* cluster_inv_ind_ptr,
              ValType* k_ptr,
              ValType* Q_arr, 
              ValType* temp_i, // size = n_edges
              ValType m2
              ){

  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  IdxType start_idx, end_idx, c_i; 
  ValType ki(0.0), Ai(0.0), sum_k(0.0);
  IdxType start_c_idx;
  IdxType end_c_idx;

  if(i < n_vertex){
    start_idx = *( csr_ptr_ptr + i );
    end_idx   = *( csr_ptr_ptr + i + 1 );

    c_i = *(cluster_ptr + i); 
    ki = *(k_ptr + i);

    //only sees its neibors
    Ai = 0.0;
#pragma unroll 
    for(int j = 0; j< end_idx - start_idx; ++j){ 
      IdxType j_idx = (IdxType)(*(csr_ind_ptr + j + start_idx));
      IdxType c_j = (IdxType)(*(cluster_ptr + j_idx));
      Ai += ((int)(c_i != c_j)*((ValType)(*(csr_val_ptr + j + start_idx))));
    }
    
    
    start_c_idx = *(cluster_inv_ptr_ptr + c_i);
    end_c_idx = *(cluster_inv_ptr_ptr + c_i + 1); 
 

#ifdef DEBUG
    if (temp_i == NULL) printf("Error in allocate temp_i memory in thread %d\n",i);
#endif

#pragma unroll
    for(int j = 0; j< end_c_idx-start_c_idx; ++j){
      IdxType j_idx = (IdxType)(*(cluster_inv_ind_ptr + j + start_c_idx));
      sum_k += (ValType)(*(k_ptr + j_idx)); 
    }     

    sum_k = m2 - sum_k;    
    *(Q_arr + i) =( Ai - (( ki * sum_k )/ m2))/m2 ;
//      printf("-- i: %d Q: %.6e Ai: %f ki*sum_k = %f x %f = %f\n", i, *(Q_arr + i), Ai, ki, sum_k, (ki * sum_k));

  }
  return;
}



template<typename IdxType=int, typename ValType> 
__device__ void
modularity_no_matrix(const int n_vertex, const int n_clusters, ValType m2, 
                     IdxType* csr_ptr_ptr, IdxType* csr_ind_ptr, ValType* csr_val_ptr, 
                     IdxType* cluster_ptr, IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr,
                     bool weighted, // bool identical_cluster, // todo  optimizaiton
                     ValType* k_vec, 
                     ValType* Q_arr, 
                     ValType* temp_i){


  compute_k_vec(n_vertex, csr_ptr_ptr, csr_val_ptr, weighted, k_vec);
  __syncthreads(); 

  modularity_i(n_vertex, n_clusters, 
               csr_ptr_ptr, csr_ind_ptr, csr_val_ptr, 
               cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
               k_vec, Q_arr, temp_i, m2);

} 



template<typename IdxType, typename ValType>
__global__ void 
kernel_modularity_no_matrix(const int n_vertex, const int n_clusters, ValType m2,
                            IdxType* csr_ptr_ptr, IdxType* csr_ind_ptr, ValType* csr_val_ptr, 
                            IdxType* cluster_ptr, IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr,
                            bool weighted, ValType* k_vec_ptr, ValType* Q_arr_ptr, ValType* temp_i_ptr){
  ValType m2_s(m2);
  modularity_no_matrix(n_vertex, n_clusters, m2_s, 
                       csr_ptr_ptr, csr_ind_ptr, csr_val_ptr, 
                       cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                       weighted, k_vec_ptr, Q_arr_ptr, temp_i_ptr );

}

template<typename IdxType, typename ValType>
ValType 
modularity(const int n_vertex, int n_edges, const int n_clusters, ValType m2,
           IdxType* csr_ptr_ptr, IdxType* csr_ind_ptr, ValType* csr_val_ptr,
           IdxType* cluster_ptr, IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr,
           bool weighted, ValType* k_vec_ptr, 
           ValType* Q_arr_ptr, ValType* temp_i_ptr // temporary space for calculation
           ){

  thrust::fill(thrust::device, temp_i_ptr, temp_i_ptr + n_edges, 0.0);

  int nthreads = min(n_vertex,CUDA_MAX_KERNEL_THREADS); 
  int nblocks = min((n_vertex + nthreads - 1)/nthreads,CUDA_MAX_BLOCKS); 
  kernel_modularity_no_matrix<<<nblocks, nthreads >>>(n_vertex, n_clusters, m2,
                                                          csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
                                                          cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                                                          weighted, k_vec_ptr, Q_arr_ptr, temp_i_ptr);

  CUDA_CALL(cudaDeviceSynchronize());

  ValType Q = thrust::reduce(thrust::cuda::par, Q_arr_ptr, Q_arr_ptr + n_vertex, (ValType)(0.0)); 

  return -Q;

} 

/***********************
cluster_iter(n_vertex)
cluster_inv_ptr(c_size + 1)
cluster_inv_ind(n_vertex)
seq_idx(n_vertex) [0, 1, 2, ... , n_vertex -1] 
***********************/
template<typename IdxIter, typename IdxType=int> 
__global__ void
generate_cluster_inv_ptr(const int n_vertex, const int c_size, IdxIter cluster_iter, IdxType* cluster_inv_ptr){
  int tid = blockDim.x * blockIdx.x + threadIdx.x; 
  IdxType ci;
  //Inital cluster_inv_ptr outside!!!

  if(tid < n_vertex){
    ci = *(cluster_iter + tid);
    atomicAdd(cluster_inv_ptr + ci, 1);
  }
}


template<typename IdxType=int, typename IdxIter> 
void
generate_cluster_inv(const int n_vertex, const int c_size, 
                    IdxIter cluster_iter, 
                    thrust::device_vector<IdxType>& cluster_inv_ptr, 
                    thrust::device_vector<IdxType>& cluster_inv_ind){

  int nthreads = min(n_vertex,CUDA_MAX_KERNEL_THREADS); 
  int nblocks = min((n_vertex + nthreads - 1)/nthreads,CUDA_MAX_BLOCKS); 
  thrust::fill(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.end(), 0);
  cudaCheckError();
  IdxType* cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());

  generate_cluster_inv_ptr<<<nblocks,nthreads>>>(n_vertex, c_size, cluster_iter, cluster_inv_ptr_ptr);
  CUDA_CALL(cudaDeviceSynchronize());

#ifdef DEBUG
  if((unsigned)c_size + 1 > cluster_inv_ptr.size())
    std::cout<<"Error cluster_inv_ptr run out of memory\n";
#endif

  thrust::exclusive_scan(thrust::device, cluster_inv_ptr.begin(), cluster_inv_ptr.begin() + c_size + 1 , cluster_inv_ptr.begin());
  cudaCheckError();

  thrust::sequence(thrust::device, cluster_inv_ind.begin(), cluster_inv_ind.end(), 0); 
  cudaCheckError();
  thrust::sort(thrust::device, cluster_inv_ind.begin(), cluster_inv_ind.begin() + n_vertex, sort_by_cluster<IdxType, IdxIter>(cluster_iter));
  cudaCheckError();  
  
}


}// nvlouvain

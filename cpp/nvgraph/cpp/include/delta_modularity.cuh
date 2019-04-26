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
#include <cuda_profiler_api.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/transform.h>

#include "util.cuh"
#include "graph_utils.cuh"
#include "functor.cuh"
//#include "block_delta_modularity.cuh"

#include <cusparse.h>


namespace nvlouvain{


/*************************************************************
*
*  compute k_i_in
*
*  - input :
*     n_vertex
*     csr_ptr's ptr 
*     csr_idx's ptr
*     csr_val's ptr
*     cluster's ptr : current cluster assignment
*     c: target cluster
*     i: current vertex
*
*  - output:
*     results: k i in c
*
***************************************************************/

template<typename IdxType, typename ValType> 
__device__ void compute_k_i_in( const int n_vertex, 
                                IdxType* csr_ptr_ptr, 
                                IdxType* csr_idx_ptr, 
                                ValType* csr_val_ptr, 
                                IdxType* cluster_ptr, 
                                IdxType c, // tid.y
                                IdxType i, // tid.x 
                                ValType* result){
  ValType sum = 0.0;
  //Sanity check
  if( i < n_vertex ){

    IdxType i_start = *(csr_ptr_ptr + i);
    IdxType i_end = *(csr_ptr_ptr + i + 1);
    
#pragma unroll    
    for(int j = 0; j < i_end - i_start; ++j){
      IdxType j_idx = *(csr_idx_ptr + i_start + j);
      IdxType   c_j = *(cluster_ptr + j_idx);
      sum += (int)(c_j==c)*((ValType)(*(csr_val_ptr + i_start + j)));
    }
    *result = sum;
  }

}


// delta modularity when an isolate vertex i moved into a cluster c 
// c must be one of the clusters 
// ptr version
template<typename IdxType, typename ValType>
__device__ void
delta_modularity(const int n_vertex, const int c_size, bool updated,
                 IdxType* csr_ptr_ptr, IdxType* csr_ind_ptr, ValType* csr_val_ptr, 
                 IdxType* cluster_ptr,
                 ValType c_sum, ValType m2,
                 IdxType row_idx, IdxType col_idx, IdxType c, ValType* k_vec_ptr, ValType* score){
 
  // ki: sum of i's edges weight 
  // ki_in: sum of edge from i to c
  // sum_tot: for all v in c, sum of v's edges weight
 
  IdxType c_i = *(cluster_ptr + row_idx);
  ValType ki_in = 0.0;
  ki_in = (int)(c_i!=c)*(*(csr_val_ptr + col_idx));
  ValType ki = *(k_vec_ptr + row_idx);
  

  if(!updated){
    compute_k_i_in(n_vertex, csr_ptr_ptr, csr_ind_ptr, csr_val_ptr, cluster_ptr, c, row_idx, &ki_in);
  }

  ValType sum_tot = c_sum - (int)(c_i == c)*ki;
  *score = ki_in - 2*sum_tot*ki/(m2);
//  printf("i: %d\tci: %d\tc: %d\t2m: %1f\tkin: %f\tki: %f\tsum_tot: %f\tc_sum: %f\tdelta: %f\n", row_idx, c_i, c, m2, ki_in, ki, sum_tot, c_sum,*score );
}



template<typename IdxType=int, typename ValType>
__device__ void compute_cluster_sum(const int n_vertex, const int c_size, 
                                    IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr, 
                                    ValType* k_ptr, // pre-compute ki size: n_vertex
                                    ValType* cluster_sum_vec){

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  IdxType c_start, c_end;
  ValType sum = 0.0;
  if(c < c_size){
    c_start = *(cluster_inv_ptr_ptr + c);
    c_end = *(cluster_inv_ptr_ptr + c + 1);  

#pragma unroll        
    for(IdxType* it = cluster_inv_ind_ptr + c_start; it!= cluster_inv_ind_ptr + c_end ; ++it){
      sum += (ValType)(*(k_ptr + *(it)));
    }
    *(cluster_sum_vec + c) = sum;
    //printf("c: %d c_sum: %f\n", c, (ValType)(*(cluster_sum_vec + c)));
  }
   

}


template<typename IdxType=int, typename ValType>
__global__ void
kernel_compute_cluster_sum(const int n_vertex, const int c_size,
                           IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr,
                           ValType* k_ptr, // pre-compute ki size: n_vertex
                           ValType* cluster_sum_vec){

  compute_cluster_sum(n_vertex, c_size, 
                      cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
                      k_ptr, cluster_sum_vec);
  
}


/****************************************************************************************************
*
*  compute delta modularity vector, delta_modularity_vec, size = n_edges
*  theads layout: (lunched as 1D)
*    1 thread for 1 edge, flattened
*    need coo row index instead (pre-computed)
*  input variables:
*    n_vertex: number of vertex
*    n_edges:  number of edges
*    c_size:   number of unique clusters
*    updated:  if previous iteration generate a new supervertices graph    
*    cluster_ptr: cluster assignment
*    cluster_sum_vec_ptr: sum of clusters
*    k_vec_ptr: ki vector 
*  output:
*    delta_modularity_vec: size = n_edges
*                          delta modularity if we move from_node to to_nodes cluster c for each edge    
*
****************************************************************************************************/
template<typename IdxType, typename ValType>
__global__ void// __launch_bounds__(CUDA_MAX_KERNEL_THREADS) 
build_delta_modularity_vec_flat(const int n_vertex, const int n_edges, const int c_size, ValType m2, bool updated,
                          IdxType* coo_row_ind_ptr, IdxType* csr_ptr_ptr,  IdxType* csr_ind_ptr, ValType* csr_val_ptr, 
                          IdxType* cluster_ptr,
                          ValType* cluster_sum_vec_ptr,
                          ValType* k_vec_ptr,
                          ValType* delta_modularity_vec){

  ValType m2_s(m2); //privatize 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if( tid < n_edges ){
    IdxType row_idx = *(coo_row_ind_ptr + tid);
    IdxType col_idx = *(csr_ind_ptr + tid);
    IdxType c = cluster_ptr[ col_idx ]; // target cluster c
    ValType c_sum = cluster_sum_vec_ptr[c]; 

    delta_modularity(n_vertex, c_size, updated,
                     csr_ptr_ptr,  csr_ind_ptr, csr_val_ptr,
                     cluster_ptr,
                     c_sum, m2_s,
                     row_idx, col_idx, c, k_vec_ptr, delta_modularity_vec + tid); 

  }
}


/******************************************************************************************************
*  NOT USED
*  compute delta modularity vector, delta_modularity_vec, size = n_edges
*  theads layout: (lauched as 2D)
*    1 thread for 1 edge 
*    each thread.x per vertex i
*    each thread.y per neibor j of vertex i 
*    need to pre compute max_degree for lauch this kernel
*  input variables:
*    n_vertex: number of vertex
*    n_edges:  number of edges
*    c_size:   number of unique clusters
*    updated:  if previous iteration generate a new supervertices graph    
*    cluster_ptr: cluster assignment
*    cluster_sum_vec_ptr: sum of clusters
*    k_vec_ptr: ki vector 
*  output:
*    delta_modularity_vec: size = n_edges
*                          delta modularity if we move from_node to to_nodes cluster c for each edge
*    
*****************************************************************************************************/
/*
template<typename IdxIter, typename ValIter, typename ValType>
__global__ void// __launch_bounds__(CUDA_MAX_KERNEL_THREADS) 
build_delta_modularity_vec(const int n_vertex, const int c_size, ValType m2, bool updated,
                          IdxIter csr_ptr_ptr, IdxIter csr_ind_ptr, ValIter csr_val_ptr, 
                          IdxIter cluster_ptr,
                          ValType* cluster_sum_vec_ptr,
                          ValType* k_vec_ptr,
                          ValType* delta_modularity_vec){

  ValType m2_s(m2);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int start, end;
  if( i < n_vertex ){

    start = *(csr_ptr_ptr + i);
    end = *(csr_ptr_ptr + i + 1);
    
    if(j < end - start){
      int j_idx = *(csr_ind_ptr + start + j);
      int c = *( cluster_ptr + j_idx);
      ValType c_sum = cluster_sum_vec_ptr[c];
      
      delta_modularity( n_vertex, c_size, updated,  
                        csr_ptr_ptr, csr_ind_ptr, csr_val_ptr, 
                        cluster_ptr, 
                        c_sum, m2_s, 
                        i, start + j, c, k_vec_ptr, delta_modularity_vec + start + j);

    }
  }
}
*/

/******************************************************
*
*  find the max delta modularity for each vertex i
*  zero out other delta modularity for vertex i
*
*******************************************************/
//template<typename ValType, typename IdxIter, typename ValIter>
template<typename ValType, typename IdxIter, typename ValIter>
__global__ void// __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
max_delta_modularity_vec_stride(const int n_vertex, const int n_edges,
                         IdxIter csr_ptr_iter, IdxIter csr_ind_iter, ValIter csr_val_iter, IdxIter cluster_iter,
                         ValType* delta_modularity_vec){
  
  unsigned int wid = blockIdx.x;  // 0 ~ n_vertex - 1
  unsigned int tid = threadIdx.x; // 0 ~ 31

  __shared__ int start_idx;
  __shared__ int end_idx;
  __shared__ int degree;
  __shared__ ValType local_max[WARP_SIZE];
  __shared__ ValType warp_max_val;
  unsigned int stride = WARP_SIZE / 2;
  warp_max_val = -1000;

  if( wid < n_vertex ){ 
    if(tid == 0){
      start_idx = *(csr_ptr_iter + wid);     
      end_idx = *(csr_ptr_iter + wid + 1);
      degree = end_idx - start_idx;      
    }
    __syncwarp();
    //find the max elements
    for(unsigned xid = 0; xid + tid < ( degree ); xid += WARP_SIZE){
      local_max[tid]= -1.0 ;

      if(start_idx + xid + tid > n_edges) 
        printf("Error access invalid memory %d = %d +  %d + %d end: %d\n", start_idx + xid + tid, start_idx, xid, tid, end_idx);

      local_max[tid] = (ValType)(*(delta_modularity_vec + start_idx + xid + tid));

      stride = umin(16, (degree)/2 + 1);
     
      while(tid < stride && stride > 0){
        local_max[tid] = fmax(local_max[tid], local_max[tid + stride]);
        
        stride/=2;  //stride /=2
      }
      __syncwarp();

      if(tid == 0 && warp_max_val < local_max[0]){
        warp_max_val = local_max[0];
      }
    } 

    __syncwarp();
    // zero out non-max elements    
    for(unsigned xid = 0; xid + tid < ( degree ); xid += WARP_SIZE){
      if(start_idx + xid + tid < end_idx){
        ValType original_val = ((ValType)*(delta_modularity_vec + start_idx + xid + tid));
        (*(delta_modularity_vec + start_idx + xid + tid)) = (int)(original_val == warp_max_val) * original_val;

/*
        if(original_val == warp_max_val){
          int j_idx =    (int)(*(csr_ind_iter + start_idx + xid + tid));
          printf("+i: %d j: %d c: %d %f\n", wid, j_idx, (int)(*(cluster_iter + j_idx)),original_val );
        }else{
          int j_idx =    (int)(*(csr_ind_iter + start_idx + xid + tid));
          printf("-i: %d j: %d c: %d %f\n", wid, j_idx, (int)(*(cluster_iter + j_idx)),original_val );
          
        }
  */    

      }
    }
    
 
  }
 
}


/******************************************************
*  NOT USED
*  find the max delta modularity for each vertex i
*  zero out other delta modularity for vertex i
*
*******************************************************/
/*
template<typename IdxIter, typename ValIter, typename ValType>
__global__ void// __launch_bounds__(CUDA_MAX_KERNEL_THREADS) 
max_delta_modularity_vec(const int n_vertex, 
                          IdxIter csr_ptr_ptr, IdxIter csr_ind_ptr, ValIter csr_val_ptr, 
                          ValType* delta_modularity_vec){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int start, end;
  ValType * best_pos_ptr;
  if( i < n_vertex ){ 
    start = *( csr_ptr_ptr + i);
    end = *( csr_ptr_ptr + i + 1);
    best_pos_ptr = thrust::max_element(thrust::cuda::par, delta_modularity_vec + start, delta_modularity_vec + end);
  }

  if( i < n_vertex ){
    //printf("i: %d max: %f\n", i, (ValType)(*best_pos_ptr));
    thrust::replace_if(thrust::cuda::par, delta_modularity_vec + start, delta_modularity_vec + end, not_best<ValType>(*best_pos_ptr), 0.0);
 
  }

}

*/
// Not used
template<typename IdxType, typename ValType>
void build_delta_modularity_vector_old(const int n_vertex, const int c_size, ValType m2, bool updated,
                                       thrust::device_vector<IdxType>& csr_ptr_d, thrust::device_vector<IdxType>& csr_ind_d, thrust::device_vector<ValType>& csr_val_d, 
                                       thrust::device_vector<IdxType>& cluster_d,
                                       IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr, // precompute cluster inverse
                                       ValType* k_vec_ptr, // precompute ki's 
                                       thrust::device_vector<ValType>& temp_vec, // temp global memory with size n_vertex
                                       ValType* cluster_sum_vec_ptr, 
                                       ValType* delta_Q_arr_ptr){

  /* start compute delta modularity vec  */
  dim3 block_size_1d((n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size_1d(BLOCK_SIZE_1D, 1, 1); 
  int n_edges = csr_ptr_d[n_vertex];
    
  kernel_compute_cluster_sum<<<block_size_1d, grid_size_1d>>>( n_vertex, c_size, 
                                                               cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                                                               k_vec_ptr, cluster_sum_vec_ptr);
  CUDA_CALL(cudaDeviceSynchronize());

  thrust::fill(thrust::cuda::par, delta_Q_arr_ptr, delta_Q_arr_ptr + n_edges, 0.0);

  //pre-compute max_degree for block_size_2D and grid_size_2D
  thrust::transform(thrust::device, csr_ptr_d.begin() + 1, csr_ptr_d.end(), csr_ptr_d.begin(), temp_vec.begin(), minus_idx<IdxType, ValType>());    
  auto max_ptr = thrust::max_element(thrust::device, temp_vec.begin(), temp_vec.begin() + n_vertex );
  int max_degree = (IdxType)(*max_ptr);

  dim3 block_size_2d((n_vertex + BLOCK_SIZE_2D*2 -1)/ (BLOCK_SIZE_2D*2), (max_degree + BLOCK_SIZE_2D -1)/ (BLOCK_SIZE_2D), 1);
  dim3 grid_size_2d(BLOCK_SIZE_2D*2, BLOCK_SIZE_2D, 1);

  // build delta modularity vec with 2D (vertex i, neighbor of i) grid size are_now(32, 16, 1)
  build_delta_modularity_vec<<<block_size_2d, grid_size_2d>>>(n_vertex, c_size, m2, updated, 
                                                              csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(),
                                                              cluster_d.begin(),
                                                              cluster_sum_vec_ptr,
                                                              k_vec_ptr, delta_Q_arr_ptr);
  CUDA_CALL(cudaDeviceSynchronize());

    
  block_size_1d = dim3((n_vertex + BLOCK_SIZE_1D*4 -1)/ BLOCK_SIZE_1D*4, 1, 1);
  grid_size_1d = dim3(BLOCK_SIZE_1D*4, 1, 1); 

  // zero out non maximum delta modularity for each vertex i grid size are now (128, 1, 1)
  max_delta_modularity_vec<<<block_size_1d, grid_size_1d>>>(n_vertex, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), delta_Q_arr_ptr );
  CUDA_CALL(cudaDeviceSynchronize());
  
}



//
// A new version of building delta modularity vector function
//  
//
template<typename IdxType, typename ValType>
void build_delta_modularity_vector(cusparseHandle_t cusp_handle, const int n_vertex, const int c_size, ValType m2, bool updated,
                                   thrust::device_vector<IdxType>& csr_ptr_d, thrust::device_vector<IdxType>& csr_ind_d, thrust::device_vector<ValType>& csr_val_d, 
                                   thrust::device_vector<IdxType>& cluster_d,
                                   IdxType* cluster_inv_ptr_ptr, IdxType* cluster_inv_ind_ptr, // precompute cluster inverse
                                   ValType* k_vec_ptr, // precompute ki's 
                                   ValType* cluster_sum_vec_ptr, 
                                   ValType* delta_Q_arr_ptr){

  /* start compute delta modularity vec  */
  dim3 block_size_1d((n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size_1d(BLOCK_SIZE_1D, 1, 1); 
  int n_edges = csr_ptr_d[n_vertex];
    
  kernel_compute_cluster_sum<<<block_size_1d, grid_size_1d>>>( n_vertex, c_size, 
                                                               cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                                                               k_vec_ptr, cluster_sum_vec_ptr);
  CUDA_CALL(cudaDeviceSynchronize());
    
  thrust::fill(thrust::cuda::par, delta_Q_arr_ptr, delta_Q_arr_ptr + n_edges, 0.0);
  IdxType *csr_ptr_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  IdxType *csr_ind_ptr = thrust::raw_pointer_cast(csr_ind_d.data()); 
  ValType *csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  IdxType *cluster_ptr = thrust::raw_pointer_cast(cluster_d.data());
  
  // pre compute coo row indices using cusparse
  thrust::device_vector<IdxType> coo_row_ind(n_edges);
  IdxType* coo_row_ind_ptr =  thrust::raw_pointer_cast(coo_row_ind.data());
  cusparseXcsr2coo(cusp_handle, csr_ptr_ptr,  
                   n_edges, n_vertex, coo_row_ind_ptr, 
                   CUSPARSE_INDEX_BASE_ZERO);  
  // build delta modularity vec flatten (1 thread per 1 edges) 
  block_size_1d = dim3((n_edges + BLOCK_SIZE_1D * 2 -1)/ BLOCK_SIZE_1D * 2, 1, 1);
  grid_size_1d  = dim3(BLOCK_SIZE_1D*2, 1, 1); 

  build_delta_modularity_vec_flat<<<block_size_1d, grid_size_1d>>>(n_vertex, n_edges, c_size, m2, updated, 
                                                                coo_row_ind_ptr, csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
                                                                cluster_ptr,
                                                                cluster_sum_vec_ptr,
                                                                k_vec_ptr, delta_Q_arr_ptr);
  CUDA_CALL(cudaDeviceSynchronize());

 // Done compute delta modularity vec
  block_size_1d = dim3(n_vertex, 1, 1);
  grid_size_1d  = dim3(WARP_SIZE, 1, 1);
 
  max_delta_modularity_vec_stride<<<block_size_1d, grid_size_1d>>>(n_vertex, n_edges, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), cluster_d.begin(), delta_Q_arr_ptr );
  CUDA_CALL(cudaDeviceSynchronize());
 

}



} // nvlouvain

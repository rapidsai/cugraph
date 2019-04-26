
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
#include "back_up.cuh"

template<typename IdxIter, typename ValIter, typename IdxType=int,  typename ValType>
__global__ void 
kernel_phase_1_color(const int n_vertex, IdxIter csr_ptr_iter, IdxIter csr_ind_iter, ValIter csr_val_iter, IdxIter cluster,  
               IdxType* color, IdxType color_size,ValType *matrix, IdxType *cluster_sizes, ValType* improve, IdxType* n_moved){



  *n_moved = 0;
  IdxType j = blockIdx.x * blockDim.x + threadIdx.x;
  IdxType i = blockIdx.y * blockDim.y + threadIdx.y;
 
  for( int t = 0; t < color_size; ++t ){ // color t
    if( i< n_vertex && color[i] == t ){
      IdxType start_idx = *(csr_ptr_iter + i);
      IdxType end_idx = *(csr_ptr_iter + i + 1);
      if(j < end_idx - start_idx){
        IdxType c = cluster[ csr_ind_iter[start_idx + j]];
        //printf("i:%d j:%d start:%d end:%d c:%d\n",i,j,start_idx, end_idx,c);
        nvlouvain::phase_1( n_vertex,
                            csr_ptr_iter,
                            csr_ind_iter,
                            csr_val_iter,
                            cluster,
                            i,
                            j,
                            c,
                            matrix, 
                            cluster_sizes, 
                            improve, n_moved);
      }
    
    }
  }
}

/*
void phase_1_color_test(thrust::device_vector<int> &csr_ptr_d,
                     thrust::device_vector<int> &csr_ind_d,
                     thrust::device_vector<T> &csr_val_d,
                     const int size){

  HighResClock hr_clock;
  double timed;
  
  dim3 block_size((size + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, (size + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, 1);
  dim3 grid_size(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1); 

 
  
  thrust::device_vector<int> cluster_d(size);
  thrust::sequence(cluster_d.begin(), cluster_d.end());  

  std::cout<<"old cluster: ";
  nvlouvain::display_vec(cluster_d);

  thrust::device_vector<T> Q_d(1);
  T* Q_d_raw_ptr = thrust::raw_pointer_cast(Q_d.data());

  thrust::device_vector<T> matrix(size*size);
  T* matrix_raw_ptr = thrust::raw_pointer_cast(matrix.data());

  hr_clock.start();

  kernel_modularity<<<block_size, grid_size>>>(size, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), cluster_d.begin(), matrix_raw_ptr, Q_d_raw_ptr);

 
  CUDA_CALL(cudaDeviceSynchronize());

  hr_clock.stop(&timed);
  double mod_time(timed);
  std::cout<<"modularity: "<<Q_d[0]<<" runtime: "<<mod_time<<std::endl;

 
  thrust::device_vector<T> improve_d(1);
  T* improve_d_raw_ptr = thrust::raw_pointer_cast(improve_d.data());
  
  thrust::device_vector<int> c_size_d(size, 1);
  int* c_size_d_raw_ptr = thrust::raw_pointer_cast(c_size_d.data());
 
  thrust::device_vector<int> n_moved(1, 0);
  int* n_moved_ptr =  thrust::raw_pointer_cast(n_moved.data());
//  nvlouvain::display_vec(c_size_d);

  //--------------------------------  1st -
  thrust::device_vector<T> Q_old(Q_d);
  double delta_Q;

  int count = 0;
  int num_move = 0;
  int color_size;
  
  std::vector<int> fill_color(size);
  if(size == 16){ 
    fill_color = {0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 0, 1, 1, 2, 1, 0};
    color_size = 3;
  }
  else if(size == 4){
    fill_color =  {0, 1, 2, 0};
    color_size = 3;
  }


  thrust::device_vector<int> color(fill_color);

  int* color_ptr = thrust::raw_pointer_cast(color.data());
  
  do{
    Q_old[0] = Q_d[0]; 
    hr_clock.start();

    kernel_phase_1_color<<<block_size, grid_size>>>(size, 
                                              csr_ptr_d.begin(), 
                                              csr_ind_d.begin(), 
                                              csr_val_d.begin(), 
                                              cluster_d.begin(), 
                                              color_ptr,
                                              color_size,
                                              matrix_raw_ptr, 
                                              c_size_d_raw_ptr,
                                              improve_d_raw_ptr, 
                                              n_moved_ptr);
  
    CUDA_CALL(cudaDeviceSynchronize());
  
    hr_clock.stop(&timed);
    mod_time = timed;
    std::cout<<"new cluster: ";
    nvlouvain::display_vec(cluster_d);  
    std::cout<<"improvement: "<<improve_d[0]<<" runtime: "<<mod_time<<std::endl;
  
    kernel_modularity<<<block_size, grid_size>>>(size, csr_ptr_d.begin(), csr_ind_d.begin(), csr_val_d.begin(), cluster_d.begin(), matrix_raw_ptr, Q_d_raw_ptr);
    CUDA_CALL(cudaDeviceSynchronize());
   

    delta_Q = Q_d[0] - Q_old[0]; 
    std::cout<<"new modularity: "<<Q_d[0]<<" delta_Q:"<<delta_Q<<" runtime: "<<mod_time<<std::endl;
    std::cout<<"cluster size: ";
    nvlouvain::display_vec(c_size_d);
 
    int sum = thrust::reduce(thrust::cuda::par, c_size_d.begin(), c_size_d.end(), 0);
    num_move = n_moved[0];
    std::cout<<"sum: "<< sum<<" moved: "<<num_move<<std::endl;
    

    ++count;
  }while(  num_move > 0 );

}
*/


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
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include "test_opt_utils.cuh"
#include "graph_utils.cuh"

//#define ENABLE_LOG TRUE
#define ENALBE_LOUVAIN true

#include "nvlouvain.cuh"
#include "gtest/gtest.h"
#include "high_res_clock.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
using T = float;

int main(int argc, char* argv[]){

  if(argc < 2)
  {
    std::cout<< "Help : ./louvain_test matrix_market_file.mtx"<<std::endl;
    return 1;
  }
  FILE* fin = std::fopen( argv[1] ,"r");
  int m, k, nnz;
  MM_typecode mc;

  CUDA_CALL(cudaSetDevice(0));

  EXPECT_EQ((mm_properties<int>(fin, 1, &mc, &m, &k, &nnz)) ,0);
  EXPECT_EQ(m,k);  

  thrust::host_vector<int> coo_ind_h(nnz);
  thrust::host_vector<int> csr_ptr_h(m+1);
  thrust::host_vector<int> csr_ind_h(nnz);
  thrust::host_vector<T> csr_val_h(nnz);

  EXPECT_EQ( (mm_to_coo<int,T>(fin, 1, nnz, &coo_ind_h[0], &csr_ind_h[0], &csr_val_h[0], NULL)), 0);
  EXPECT_EQ( (coo_to_csr<int,T> (m, k, nnz, &coo_ind_h[0], &csr_ind_h[0], &csr_val_h[0], NULL, &csr_ptr_h[0], NULL, NULL, NULL)), 0);

  EXPECT_EQ(fclose(fin),0); 

  thrust::device_vector<int> csr_ptr_d(csr_ptr_h);
  thrust::device_vector<int> csr_ind_d(csr_ind_h);
  thrust::device_vector<T> csr_val_d(csr_val_h);
  
  thrust::device_vector<T> tmp_1(nnz);
  thrust::fill(thrust::cuda::par, tmp_1.begin(), tmp_1.end(), 1.0);
  thrust::device_vector<T>::iterator max_ele = thrust::max_element(thrust::cuda::par, csr_val_d.begin(), csr_val_d.end());

  bool weighted = (*max_ele!=1.0);

  //std::cout<<(weighted?"Weighted ":"Not Weigthed ")<<" n_vertex: "<<m<<"\n";

  HighResClock hr_clock;
  double louvain_time;
  if(ENALBE_LOUVAIN){
    T final_modulartiy(0);    
    //bool record = true;
    bool has_init_cluster = false;
    int *clustering_h = (int*)malloc(m*sizeof(int));
    thrust::device_vector<int> cluster_d(m, 0);
    int* csr_ptr_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
    int* csr_ind_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
    T* csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());      
    int* init_cluster_ptr = thrust::raw_pointer_cast(cluster_d.data());
    int num_level;
    
    cudaProfilerStart(); 
    hr_clock.start(); 
    nvlouvain::louvain<int,T>(csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
                            m, nnz, 
                            weighted, has_init_cluster,  
                            init_cluster_ptr, final_modulartiy, clustering_h, num_level);

    hr_clock.stop(&louvain_time);
    cudaProfilerStop();

    std::cout<<"Final modularity: "<<COLOR_MGT<<final_modulartiy<<COLOR_WHT<<" num_level: "<<num_level<<std::endl;
    std::cout<<"louvain total runtime:"<<louvain_time/1000<<" ms\n"; 
  }
  return 0;
}


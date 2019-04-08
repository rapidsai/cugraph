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
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <string>
#include <time.h>

namespace nvlouvain{

#define BLOCK_SIZE_1D 64
#define BLOCK_SIZE_2D 16
#define CUDA_MAX_KERNEL_THREADS 256
#define CUDA_MAX_BLOCKS_1D 65535
#define CUDA_MAX_BLOCKS_2D 256
#define LOCAL_MEM_MAX 512
#define GRID_MAX_SIZE 65535
#define WARP_SIZE 32

#define CUDA_CALL( call )                                                                         \
{                                                                                                 \
  cudaError_t cudaStatus = call;                                                                  \
  if ( cudaSuccess != cudaStatus )                                                                \
    fprintf(stderr, "ERROR: CUDA call \"%s\" in line %d of file %s failed with %s (%d).\n",       \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);   \
}

#define THRUST_SAFE_CALL( call )                                                                  \
{                                                                                                 \
  try{                                                                                            \
    call;                                                                                         \
  }                                                                                               \
  catch(std::bad_alloc &e){                                                                       \
    fprintf(stderr, "ERROR: THRUST call \"%s\".\n"                                                \
                      #call);                                                                     \
    exit(-1);                                                                                     \
  }                                                                                               \
} 

#define COLOR_GRN "\033[0;32m"
#define COLOR_MGT "\033[0;35m"
#define COLOR_WHT "\033[0;0m"

inline std::string time_now(){ 
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  char buff[100];
  strftime(buff, sizeof buff, "%T", gmtime(&ts.tv_sec));
  std::string s = buff;
  s +="."+std::to_string(ts.tv_nsec).substr(0, 6);

  return s;
}

typedef enum{
  NVLOUVAIN_OK = 0,
  NVLOUVAIN_ERR_BAD_PARAMETERS = 1,
}NVLOUVAIN_STATUS;

using nvlouvainStatus_t = NVLOUVAIN_STATUS;

const char* nvlouvainStatusGetString(nvlouvainStatus_t status){
  std::string s;
  switch(status){
    case 0:
      s = "NVLOUVAIN_OK";
    break;
    case 1:
      s = "NVLOUVAIN_ERR_BAD_PARAMETERS";
    break;
    default:
    break;
  }
  return s.c_str();
}

template<typename VecType> 
void display_vec(VecType vec, std::ostream& ouf=std::cout){
  auto it = vec.begin();
  ouf<<vec.front();
  for(it = vec.begin() + 1; it!= vec.end(); ++it) {
    ouf<<", "<<(*it);
  }
  ouf<<"\n";
}

template<typename VecType> 
void display_intvec_size(VecType vec, unsigned size){
  printf("%d", (int)vec[0]);
  for(unsigned i = 1; i < size; ++i) {
    printf(", %d",(int)vec[i]);
  }
  printf("\n");
}


template<typename VecType> 
void display_vec_size(VecType vec, unsigned size){
  for(unsigned i = 0; i < size; ++i) {
    printf("%f ",vec[i]);
  }
  printf("\n");
}

template<typename VecIter> 
__host__ __device__ void display_vec(VecIter vec, int size){
  
  for(unsigned i = 0; i < size; ++i) {
    printf("%f ", (*(vec+i)));
  }
  printf("\n");
}


template<typename VecType> 
__host__ __device__ void display_vec_with_idx(VecType vec, int size, int offset=0){
  
  for(unsigned i = 0; i < size; ++i) {
    printf("idx:%d %f\n", i+offset, (*(vec+i)));
  }
  printf("\n");
}

template<typename VecType> 
void display_cluster(std::vector<VecType>& vec, std::ostream& ouf=std::cout){
  
  for(const auto& it: vec){
    for(unsigned idx = 0; idx <it.size(); ++idx ){
      ouf<<idx<<" "<<it[idx]<<std::endl;
    }
  }
}

template<typename VecType>
int folded_print_float(VecType s){
  return printf("%f\n", s);
}

template<typename VecType1, typename ... VecType2>
int folded_print_float(VecType1 s, VecType2 ... vec){
  return printf("%f ", s) + folded_print_float(vec...);
}


template<typename VecType>
int folded_print_int(VecType s){
  return printf("%d\n", (int)s);
}

template<typename VecType1, typename ... VecType2>
int folded_print_int(VecType1 s, VecType2 ... vec){
  return printf("%d ", (int)s) + folded_print_int(vec...);
}

}//nvlouvain

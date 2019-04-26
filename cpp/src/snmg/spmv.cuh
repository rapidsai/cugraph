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

// snmg spmv
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
#include <omp.h>
#include "graph_utils.cuh"
#include "snmg_utils.cuh"
//#define SNMG_DEBUG

namespace cugraph
{

template <typename idx_t,typename val_t>
gdf_error snmg_csrmv (SNMGinfo & env, size_t* part_off, idx_t * off, idx_t * ind, val_t * val, val_t ** x) {
  sync_all();
  void* cub_d_temp_storage = NULL;
  size_t cub_temp_storage_bytes = 0;
  cudaStream_t stream{nullptr};
  auto i = env.get_thread_num();
  auto p = env.get_num_threads(); 
  size_t v_glob = part_off[p];
  size_t v_loc = part_off[i+1]-part_off[i];
  idx_t tmp;
  CUDA_TRY(cudaMemcpy(&tmp, &off[v_loc], sizeof(idx_t),cudaMemcpyDeviceToHost));
  size_t e_loc = tmp;
  val_t* y_loc;
  //double t = omp_get_wtime();
  
  // Allocate the local result
  ALLOC_MANAGED_TRY ((void**)&y_loc, v_loc*sizeof(val_t), stream);

  // get temporary storage size for CUB
  CUDA_TRY(cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, 
  	                              val, off, ind, x[i], y_loc, v_loc, v_glob, e_loc));
  // Allocate CUB's temporary storage
  ALLOC_MANAGED_TRY ((void**)&cub_d_temp_storage, cub_temp_storage_bytes, stream);

  // Local SPMV
  CUDA_TRY(cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, 
  	                              val, off, ind, x[i], y_loc, v_loc, v_glob, e_loc));
  print_mem_usage();	
  // Free CUB's temporary storage
  ALLOC_FREE_TRY(cub_d_temp_storage, stream);
  //#pragma omp master 
  //{std::cout <<  omp_get_wtime() - t << " ";}

  // Wait for all local spmv
  //t = omp_get_wtime();
  sync_all();
  //#pragma omp master 
  //{std::cout <<  omp_get_wtime() - t << " ";}

  //Update the output vector
  allgather (env, part_off, y_loc, x);

  return GDF_SUCCESS;
}

} //namespace cugraph

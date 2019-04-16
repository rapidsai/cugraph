/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// snmg spmv
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
#include <omp.h>
//#define SNMG_DEBUG

namespace cugraph
{

void sync_all() {
  cudaDeviceSynchronize();
  #pragma omp barrier 
}

template <typename val_t>
gdf_error csrmv_allgather (size_t* part_off, val_t* y_loc, val_t ** x) {
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads();  
  size_t v_loc= part_off[i+1]-part_off[i];
  // send the local spmv output (y_loc) to all peers to reconstruct the global vector x 
  // After this call each peer has a full, updated, copy of x
  for (int j = 0; j < p; ++j)
    CUDA_TRY(cudaMemcpy(x[j]+part_off[i], y_loc, v_loc*sizeof(val_t),cudaMemcpyDeviceToDevice));
  
  //Make sure everyone has finished copying before returning
  sync_all();

  return GDF_SUCCESS;
}

template <typename idx_t,typename val_t>
gdf_error snmg_csrmv (size_t* part_off, idx_t * off, idx_t * ind, val_t * val, val_t ** x) {
  sync_all();
  void* cub_d_temp_storage = NULL;
  size_t cub_temp_storage_bytes = 0;
  cudaStream_t stream{nullptr};
  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads(); 
  size_t v_glob = part_off[p];
  size_t v_loc = part_off[i+1]-part_off[i];
  idx_t tmp;
  CUDA_TRY(cudaMemcpy(&tmp, &off[v_loc], sizeof(idx_t),cudaMemcpyDeviceToHost));
  size_t e_loc = tmp;
  val_t* y_loc;

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
  // Free CUB's temporary storage
  ALLOC_FREE_TRY(cub_d_temp_storage, stream);

  // Wait for all local spmv
  sync_all();

  //Update the output vector
  csrmv_allgather (part_off, y_loc, x);

  return GDF_SUCCESS;
}

} //namespace cugraph

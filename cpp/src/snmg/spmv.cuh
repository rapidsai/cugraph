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

// Pagerank solver
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include "cub/cub.cuh"
//#include "nccl.h"
#include <omp.h>

#define SNMG_VERBOSE

namespace cugraph
{


template <typename idx_t,typename val_t>
gdf_error snmg_csrmv (size_t* part_off, size_t v_glob, size_t e_loc, idx_t * off, idx_t * ind, val_t * val, val_t ** x) {
  void*    cub_d_temp_storage = NULL;
  size_t   cub_temp_storage_bytes = 0;
  cudaStream_t stream{nullptr};

  auto i = omp_get_thread_num();
  auto p = omp_get_num_threads(); 
  size_t v_loc= part_off[i+1]-part_off[i];

  #ifdef SNMG_VERBOSE
  #pragma omp barrier 
  #pragma omp master 
  { 
    std::cout << v_loc << std::endl;
    std::cout << v_glob << std::endl;
    std::cout << e_loc << std::endl;
    //printv(v_loc+1,off,0);
    //printv(e_loc, ind, 0);
  }
  #pragma omp barrier 
  #endif

  val_t* y_loc;
  ALLOC_MANAGED_TRY ((void**)&y_loc, v_loc*sizeof(val_t), stream);
  // get temporary storage size
  CUDA_TRY(cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, 
  	                              val, off, ind, x[i], y_loc, v_loc, v_glob, e_loc));

  // Allocate temporary storage
  ALLOC_MANAGED_TRY ((void**)&cub_d_temp_storage, cub_temp_storage_bytes, stream);

  // SPMV
  CUDA_TRY(cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, 
  	                              val, off, ind, x[i], y_loc, v_loc, v_glob, e_loc));

  cudaCheckError()

  for (int j = 0; j < p; ++j)
    if (i != j)
        //CUDA_TRY(cudaMemcpy(x[j]+part_off[i], y_loc, v_loc*sizeof(val_t),cudaMemcpyDeviceToDevice));
        CUDA_TRY(cudaMemcpyPeer(x[j]+part_off[i],j, y_loc,i, v_loc*sizeof(val_t)));

  ALLOC_FREE_TRY(cub_d_temp_storage, stream);

  return GDF_SUCCESS;
}

} //namespace cugraph

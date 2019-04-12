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
namespace cugraph
{
template <typename idx_t,typename val_t>
gdf_error snmg_csrmv_impl (size_t v, size_t e, idx_t * off, idx_t * ind, val_t * val, val_t * x, val_t * y) {
//  void*    cub_d_temp_storage = NULL;
//  size_t   cub_temp_storage_bytes = 0;
//  // get temporary storage size
//  cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, cscVal,
//                                           cscPtr, cscInd, tmp, pagerank_vector, n, n, e);
//  // Allocate temporary storage
//  ALLOC_MANAGED_TRY ((void**)&cub_d_temp_storage, cub_temp_storage_bytes, stream);
//
//  //SPMV
//  cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, cscVal,
//      cscPtr, cscInd, tmp, pr,
//      n, n, e);
//  ALLOC_FREE_TRY(cub_d_temp_storage, stream);
//
//  //Update vector
//  //TODO
  return GDF_SUCCESS;
}

} //namespace cugraph

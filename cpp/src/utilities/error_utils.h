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
/** ---------------------------------------------------------------------------*
 * @brief Error utilities
 *
 * @file error_utils.h
 * ---------------------------------------------------------------------------**/

#ifndef GDF_ERRORUTILS_H
#define GDF_ERRORUTILS_H

#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "nvgraph_error_utils.h"

#include <cudf/types.h>

#define cudaCheckError() {                                                  \
    cudaError_t e=cudaGetLastError();                                       \
    if(e!=cudaSuccess) {                                                    \
      std::cerr << "Cuda failure: "  << cudaGetErrorString(e) << " at: "    \
        << __FILE__ << ':' << __LINE__ << std::endl;                        \
    }                                                                       \
  }

#define CUDA_TRY( call ) 									                \
{                                                                           \
    cudaError_t cudaStatus = call;                                          \
    if ( cudaSuccess != cudaStatus )                                        \
    {                                                                       \
        std::cerr << "ERROR: CUDA Runtime call " << #call                   \
                  << " in line " << __LINE__                                \
                  << " of file " << __FILE__                                \
                  << " failed with " << cudaGetErrorString(cudaStatus)      \
                  << " (" << cudaStatus << ").\n";                          \
        return GDF_CUDA_ERROR;                          				    \
    }												                        \
}                                                                                                  

#define RMM_TRY(x)  if ((x)!=RMM_SUCCESS) return GDF_MEMORYMANAGER_ERROR;

#define RMM_TRY_CUDAERROR(x)  if ((x)!=RMM_SUCCESS) return cudaPeekAtLastError();

#define CUDA_CHECK_LAST() CUDA_TRY(cudaPeekAtLastError())

#define GDF_TRY(x) 				\
{							    \
gdf_error err_code = (x);       \
if (err_code != GDF_SUCCESS)    \
	return err_code;	        \
}

#define GDF_REQUIRE(F, S) if (!(F)) return (S);

#endif // GDF_ERRORUTILS_H

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

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cudf/types.h>
#include "nvgraph_error_utils.h"
#include <thrust/sort.h>

namespace cugraph { 
namespace detail {

// Function for checking 0-based indexing
template <typename T>
void indexing_check (T* srcs, T* dests, int64_t nnz) {
#if 0
    cudaStream_t stream {nullptr};

    // min from srcs 
    auto minId_it = thrust::min_element(rmm::exec_policy(stream)->on(stream), srcs, srcs + nnz);
    T minId;
    CUDA_TRY(cudaMemcpy(&minId, minId_it, sizeof(T), cudaMemcpyDefault));
    // negative index are not allowed
    if (minId < 0 )
        return GDF_INVALID_API_CALL; 
    
    minId_it = thrust::min_element(rmm::exec_policy(stream)->on(stream), dests, dests + nnz);
    T minId2;
    CUDA_TRY(cudaMemcpy(&minId2, minId_it, sizeof(T), cudaMemcpyDefault));        
    // negative index are not allowed
    if (minId2 < 0 )
        return GDF_INVALID_API_CALL; 

    minId = minId < minId2 ? minId : minId2;

    // warning when smallest vertex is not 0
    if (minId > 0 ) {
        std::cerr<< "WARNING: the smallest vertex identifier in the edge set is "<<minId<<". ";
        std::cerr<< "Cugraph supports 0-based indexing. ";
        std::cerr<< "Hence, the smallest vertex is assumed to be 0" << std::endl;
        std::cerr<< "Vertex [0, "<<  minId <<") will be created."<< std::endl;
        std::cerr<< "If this is not intended, please refer to ";
        std::cerr<< "cuGraph renumbering feature." << std::endl;
    }
#endif
    
} 

} } //namespace

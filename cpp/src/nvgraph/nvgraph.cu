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

#include <nvgraph/nvgraph.h>   // public header **This is NVGRAPH C API**

#include "include/nvlouvain.cuh"
#include "include/nvgraph_error.hxx"

nvgraphStatus_t NVGRAPH_API nvgraphLouvain (cudaDataType_t index_type, cudaDataType_t val_type, const size_t num_vertex, const size_t num_edges,
                            void* csr_ptr, void* csr_ind, void* csr_val, int weighted, int has_init_cluster, void* init_cluster,
                            void* final_modularity, void* best_cluster_vec, void* num_level, int max_iter)
{
    NVLOUVAIN_STATUS status = NVLOUVAIN_OK;
    if ((csr_ptr == NULL) || (csr_ind == NULL) || ((csr_val == NULL) && (weighted == 1)) ||
        ((init_cluster == NULL) && (has_init_cluster == 1)) || (final_modularity == NULL) || (best_cluster_vec == NULL) || (num_level == NULL))
       return NVGRAPH_STATUS_INVALID_VALUE;

    std::ostream log(0);
    bool weighted_b = weighted;
    bool has_init_cluster_b = has_init_cluster;
    if (val_type == CUDA_R_32F)
        status = nvlouvain::louvain ((int*)csr_ptr, (int*)csr_ind, (float*)csr_val, num_vertex, num_edges,
               weighted_b, has_init_cluster_b, (int*)init_cluster, *((float*)final_modularity),
              (int*)best_cluster_vec,*((int*)num_level), max_iter, log);
    else
        status = nvlouvain::louvain ((int*)csr_ptr, (int*)csr_ind, (double*)csr_val, num_vertex, num_edges,
                weighted_b, has_init_cluster_b, (int*)init_cluster, *((double*)final_modularity),
                (int*)best_cluster_vec,*((int*)num_level), max_iter, log);

    if (status != NVLOUVAIN_OK)
        return NVGRAPH_STATUS_INTERNAL_ERROR;

    return NVGRAPH_STATUS_SUCCESS;
}

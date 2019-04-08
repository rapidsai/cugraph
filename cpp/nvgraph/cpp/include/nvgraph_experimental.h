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

// Internal header of NVGRAPH library
//
//
// WARNING:
// This header give access to experimental feature and internal routines that are not in the official API
//
//
#include "nvgraph.h"


#ifdef __cplusplus
#include "cstdio"
#else
#include "stdio.h"
#endif

#ifndef NVGRAPH_API
#ifdef _WIN32
#define NVGRAPH_API __stdcall
#else
#define NVGRAPH_API 
#endif
#endif

#ifdef __cplusplus
  extern "C" {
#endif

/* Edge matching types */
typedef enum
{
   NVGRAPH_UNSCALED  = 0, // using edge values as is
   NVGRAPH_SCALED_BY_ROW_SUM   = 1,  // 0.5*(A_ij+A_ji)/max(d(i),d (j)), where d(i) is the sum of the row i
   NVGRAPH_SCALED_BY_DIAGONAL   = 2,  // 0.5*(A_ij+A_ji)/max(diag(i),diag(j)) 
} nvgraphEdgeWeightMatching_t;
 

nvgraphStatus_t NVGRAPH_API nvgraphSpectralModularityMaximization(nvgraphHandle_t handle, 
                                   const nvgraphGraphDescr_t graph_descr, 
                                   const size_t weight_index,
                                   const int n_clusters, 
                                   const int n_eig_vects,
                                   const float evs_tolerance,
                                   const int evs_max_iter,
                                   const float kmean_tolerance,
                                   const int kmean_max_iter,
                                   int* clustering,
                                   void* eig_vals,
                                   void* eig_vects); 

nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeModularityClustering(nvgraphHandle_t handle, 
                                             const nvgraphGraphDescr_t graph_descr, 
                                             const size_t weight_index,
                                             const int clusters,
                                             const int* clustering,
                                             float * modularity);

nvgraphStatus_t NVGRAPH_API nvgraphHeavyEdgeMatching(nvgraphHandle_t handle, 
                                             const nvgraphGraphDescr_t graph_descr, 
                                             const size_t weight_index,
                                             const nvgraphEdgeWeightMatching_t similarity_metric,
                                             int* aggregates,
                                             size_t* n_aggregates);

nvgraphStatus_t NVGRAPH_API nvgraphBalancedCutClustering(nvgraphHandle_t handle, 
                                   const nvgraphGraphDescr_t graph_descr, 
                                   const size_t weight_index,
                                   const int n_clusters, 
                                   const int n_eig_vects,
                                   const int evs_type,
                                   const float evs_tolerance,
                                   const int evs_max_iter,
                                   const float kmean_tolerance,
                                   const int kmean_max_iter,
                                   int* clustering,
                                   void* eig_vals,
                                   void* eig_vects); 

nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeBalancedCut(nvgraphHandle_t handle, 
                                             const nvgraphGraphDescr_t graph_descr, 
                                             const size_t weight_index,
                                             const int n_clusters,
                                             const int* clustering,
                                             float * edgeCut, 
                                             float * ratioCut);

nvgraphStatus_t NVGRAPH_API nvgraphKrylovPagerank(nvgraphHandle_t handle, 
                                   const nvgraphGraphDescr_t graph_descr, 
                                   const size_t weight_index,
                                   const void *alpha, 
                                   const size_t bookmark_index,                                   
                                   const float tolerance, 
                                   const int max_iter, 
                                   const int subspace_size, 
                                   const int has_guess,
                                   const size_t pagerank_index); 

#if defined(__cplusplus) 
} //extern "C"
#endif


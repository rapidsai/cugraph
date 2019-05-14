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
#include <cugraph.h>
#include "utilities/error_utils.h"
#include "utilities/graph_utils.cuh"

gdf_error gdf_degree_impl(int n, int e, gdf_column* col_ptr, gdf_column* degree, bool offsets) {
  if(offsets == true) {
    dim3 nthreads, nblocks;
    nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    switch (col_ptr->dtype) {
      case GDF_INT32:   cugraph::degree_offsets<int32_t, float> <<<nblocks, nthreads>>>(n, e, static_cast<int*>(col_ptr->data), static_cast<int*>(degree->data));break;
      default: return GDF_UNSUPPORTED_DTYPE;
    }
  }
  else {
    dim3 nthreads, nblocks;
    nthreads.x = min(e, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    switch (col_ptr->dtype) {
      case GDF_INT32:   cugraph::degree_coo<int32_t, float> <<<nblocks, nthreads>>>(n, e, static_cast<int*>(col_ptr->data), static_cast<int*>(degree->data));break;
      default: return GDF_UNSUPPORTED_DTYPE;
    }
  }
  return GDF_SUCCESS;
}


gdf_error gdf_degree(gdf_graph *graph, gdf_column *degree, int x) {
  // Calculates the degree of all vertices of the graph
  // x = 0: in+out degree
  // x = 1: in-degree
  // x = 2: out-degree
  GDF_REQUIRE(graph->adjList != nullptr || graph->transposedAdjList != nullptr, GDF_INVALID_API_CALL);
  int n;
  int e;
  if(graph->adjList != nullptr) {
    n = graph->adjList->offsets->size -1;
    e = graph->adjList->indices->size;
  }
  else {
    n = graph->transposedAdjList->offsets->size - 1;
    e = graph->transposedAdjList->indices->size;
  }

  if(x!=1) {
    // Computes out-degree for x=0 and x=2
    if(graph->adjList)
      gdf_degree_impl(n, e, graph->adjList->offsets, degree, true);
    else
      gdf_degree_impl(n, e, graph->transposedAdjList->indices, degree, false);
  }

  if(x!=2) {
    // Computes in-degree for x=0 and x=1
    if(graph->adjList)
      gdf_degree_impl(n, e, graph->adjList->indices, degree, false);
    else
      gdf_degree_impl(n, e, graph->transposedAdjList->offsets, degree, true);
  }
  return GDF_SUCCESS;
}

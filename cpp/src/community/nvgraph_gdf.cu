// -*-c++-*-

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
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_gdf.cu
 * ---------------------------------------------------------------------------**/

#include <cugraph.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include <ctime>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"
#include <rmm_utils.h>

namespace cugraph {

void louvain(Graph *graph, void *final_modularity, void *num_level, void *louvain_parts_ptr, int max_iter) {

  CHECK_GRAPH(graph);

  size_t n = graph->adjList->offsets->size - 1;
  size_t e = graph->adjList->indices->size;

  void* offsets_ptr = graph->adjList->offsets->data;
  void* indices_ptr = graph->adjList->indices->data;

  void* value_ptr;
  rmm::device_vector<float> d_values;
  if(graph->adjList->edge_data) {
      value_ptr = graph->adjList->edge_data->data;
  }
  else {
      cudaStream_t stream {nullptr};
      d_values.resize(graph->adjList->indices->size);
      thrust::fill(rmm::exec_policy(stream)->on(stream), d_values.begin(), d_values.end(), 1.0);
      value_ptr = (void * ) thrust::raw_pointer_cast(d_values.data());
  }

  auto gdf_to_cudadtype= [](gdf_column *col){
    cudaDataType_t cuda_dtype;
    switch(col->dtype){
      case GDF_INT8: cuda_dtype = CUDA_R_8I; break;
      case GDF_INT32: cuda_dtype = CUDA_R_32I; break;
      case GDF_FLOAT32: cuda_dtype = CUDA_R_32F; break;
      case GDF_FLOAT64: cuda_dtype = CUDA_R_64F; break;
      default: throw new std::invalid_argument("Cannot convert data type");
      }return cuda_dtype;
  };

  cudaDataType_t index_type = gdf_to_cudadtype(graph->adjList->indices);
  cudaDataType_t val_type = graph->adjList->edge_data? gdf_to_cudadtype(graph->adjList->edge_data): CUDA_R_32F;

  nvgraphLouvain(index_type, val_type, n, e, offsets_ptr, indices_ptr, value_ptr, 1, 0, NULL,
                 final_modularity, louvain_parts_ptr, num_level, max_iter);
  
}

} //namespace cugraph

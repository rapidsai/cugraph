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

void balancedCutClustering_nvgraph(Graph* gdf_G,
                                            const int num_clusters,
                                            const int num_eigen_vects,
                                            const float evs_tolerance,
                                            const int evs_max_iter,
                                            const float kmean_tolerance,
                                            const int kmean_max_iter,
                                            gdf_column* clustering) {

  CUGRAPH_EXPECTS(gdf_G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering->data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(!clustering->valid, "Column must be valid");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
        cudaDataType_t settype;
        rmm::device_vector<double> d_val;

  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false);
  int weight_index = 0;

  cudaStream_t stream{nullptr};

  if (gdf_G->adjList->edge_data == nullptr) {
    // use a fp64 vector  [1,...,1]
    settype = CUDA_R_64F;
    d_val.resize(gdf_G->adjList->indices->size);
    thrust::fill(rmm::exec_policy(stream)->on(stream), d_val.begin(), d_val.end(), 1.0);
    NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                  nvgraph_G,
                                  weight_index,
                                  settype,
                                  (void * ) thrust::raw_pointer_cast(d_val.data())));
  }
  else {
    switch (gdf_G->adjList->edge_data->dtype) {
      case GDF_FLOAT32:
        settype = CUDA_R_32F;
        break;
      case GDF_FLOAT64:
        settype = CUDA_R_64F;
        break;
      default:
        CUGRAPH_FAIL("Unsupported data type");
    }
  }


  // Pack parameters for call to Nvgraph
  SpectralClusteringParameter param;
  param.n_clusters = num_clusters;
  param.n_eig_vects = num_eigen_vects;
  param.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS;
  param.evs_tolerance = evs_tolerance;
  param.evs_max_iter = evs_max_iter;
  param.kmean_tolerance = kmean_tolerance;
  param.kmean_max_iter = kmean_max_iter;

  // Make call to Nvgraph balancedCutClustering
  void* eig_vals = malloc(num_eigen_vects * sizeof(double));
  void* eig_vects = malloc(num_eigen_vects * clustering->size * sizeof(double));
  nvgraphStatus_t err = nvgraphSpectralClustering(nvg_handle,
                                                  nvgraph_G,
                                                  weight_index,
                                                  &param,
                                                  (int*) clustering->data,
                                                  eig_vals,
                                                  eig_vects);
  free(eig_vals);
  free(eig_vects);
  NVG_TRY(err);
  NVG_TRY(nvgraphDestroyGraphDescr(nvg_handle, nvgraph_G));
  NVG_TRY(nvgraphDestroy(nvg_handle));
  
}

void spectralModularityMaximization_nvgraph(Graph* gdf_G,
                                                      const int n_clusters,
                                                      const int n_eig_vects,
                                                      const float evs_tolerance,
                                                      const int evs_max_iter,
                                                      const float kmean_tolerance,
                                                      const int kmean_max_iter,
                                                      gdf_column* clustering) {

  CUGRAPH_EXPECTS(gdf_G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering->data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(!clustering->valid, "Column must be valid");

  // Ensure that the input graph has values
  CUGRAPH_EXPECTS(gdf_G->adjList->edge_data != nullptr, "Invalid API parameter");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false);
  int weight_index = 0;

  // Pack parameters for call to Nvgraph
  SpectralClusteringParameter param;
  param.n_clusters = n_clusters;
  param.n_eig_vects = n_eig_vects;
  param.algorithm = NVGRAPH_MODULARITY_MAXIMIZATION;
  param.evs_tolerance = evs_tolerance;
  param.evs_max_iter = evs_max_iter;
  param.kmean_tolerance = kmean_tolerance;
  param.kmean_max_iter = kmean_max_iter;

  // Make call to Nvgraph balancedCutClustering
  void* eig_vals = malloc(n_eig_vects * sizeof(double));
  void* eig_vects = malloc(n_eig_vects * clustering->size * sizeof(double));
  nvgraphStatus_t err = nvgraphSpectralClustering(nvg_handle,
                                                  nvgraph_G,
                                                  weight_index,
                                                  &param,
                                                  (int*) clustering->data,
                                                  eig_vals,
                                                  eig_vects);
  free(eig_vals);
  free(eig_vects);
  NVG_TRY(err);
  NVG_TRY(nvgraphDestroyGraphDescr(nvg_handle, nvgraph_G));
  NVG_TRY(nvgraphDestroy(nvg_handle));
  
}

void analyzeClustering_modularity_nvgraph(Graph* gdf_G,
                                                    const int n_clusters,
                                                    gdf_column* clustering,
                                                    float* score) {

  CUGRAPH_EXPECTS(gdf_G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList->edge_data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering->data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(!clustering->valid, "Column must be valid");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false);
  int weight_index = 0;

  // Make Nvgraph call

  NVG_TRY(nvgraphAnalyzeClustering(nvg_handle,
                                    nvgraph_G,
                                    weight_index,
                                    n_clusters,
                                    (const int* )clustering->data,
                                    NVGRAPH_MODULARITY,
                                    score));
  
}

void analyzeClustering_edge_cut_nvgraph(Graph* gdf_G,
                                                  const int n_clusters,
                                                  gdf_column* clustering,
                                                  float* score) {

  CUGRAPH_EXPECTS(gdf_G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering->data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(!clustering->valid, "Column must be valid");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
        cudaDataType_t settype;
        rmm::device_vector<double> d_val;

  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false);
  int weight_index = 0;

  cudaStream_t stream{nullptr};

  if (gdf_G->adjList->edge_data == nullptr) {
    // use a fp64 vector  [1,...,1]
    settype = CUDA_R_64F;
    d_val.resize(gdf_G->adjList->indices->size);
    thrust::fill(rmm::exec_policy(stream)->on(stream), d_val.begin(), d_val.end(), 1.0);
    NVG_TRY(nvgraphAttachEdgeData(nvg_handle,
                                  nvgraph_G,
                                  weight_index,
                                  settype,
                                  (void * ) thrust::raw_pointer_cast(d_val.data())));
  }
  else {
    switch (gdf_G->adjList->edge_data->dtype) {
      case GDF_FLOAT32:
        settype = CUDA_R_32F;
        break;
      case GDF_FLOAT64:
        settype = CUDA_R_64F;
        break;
      default:
        CUGRAPH_FAIL("Unsupported data type");
      }
  }

  // Make Nvgraph call

  NVG_TRY(nvgraphAnalyzeClustering(nvg_handle,
                                    nvgraph_G,
                                    weight_index,
                                    n_clusters,
                                    (const int* )clustering->data,
                                    NVGRAPH_EDGE_CUT,
                                    score));
  
}

void analyzeClustering_ratio_cut_nvgraph(Graph* gdf_G,
                                                  const int n_clusters,
                                                  gdf_column* clustering,
                                                  float* score) {

  CUGRAPH_EXPECTS(gdf_G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList->edge_data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(clustering->data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(!clustering->valid, "Column must be valid");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false);
  int weight_index = 0;

  // Make Nvgraph call

  NVG_TRY(nvgraphAnalyzeClustering(nvg_handle,
                                    nvgraph_G,
                                    weight_index,
                                    n_clusters,
                                    (const int* )clustering->data,
                                    NVGRAPH_RATIO_CUT,
                                    score));
  
}


void extract_subgraph_vertex_nvgraph(Graph* gdf_G,
                                              gdf_column* vertices,
                                              Graph* result) {

  CUGRAPH_EXPECTS(gdf_G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(gdf_G->adjList != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(vertices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(vertices->data != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(!vertices->valid, "Column must be valid");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvg_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, gdf_G, &nvg_G, false);

  // Create an Nvgraph graph descriptor for the result and initialize
  nvgraphGraphDescr_t nvg_result = nullptr;
  NVG_TRY(nvgraphCreateGraphDescr(nvg_handle, &nvg_result));

  // Call Nvgraph function to get subgraph (into nv_result descriptor)
  NVG_TRY(nvgraphExtractSubgraphByVertex(nvg_handle,
					 nvg_G,
					 nvg_result,
					 (int*)vertices->data,
					 vertices->size));

  // Get the vertices and edges of the created subgraph to allocate memory:
  nvgraphCSRTopology32I_st topo;
  topo.source_offsets = nullptr;
  topo.destination_indices = nullptr;
  nvgraphTopologyType_t TT = NVGRAPH_CSR_32;
  NVG_TRY(nvgraphGetGraphStructure(nvg_handle, nvg_result, (void*)&topo, &TT));
  if (TT != NVGRAPH_CSR_32)
    CUGRAPH_FAIL("Unsupported nvgraph topology");
  int num_verts = topo.nvertices;
  int num_edges = topo.nedges;
  result->adjList = new gdf_adj_list;
  result->adjList->offsets = new gdf_column;
  result->adjList->indices = new gdf_column;
  result->adjList->ownership = 0;
  int *offsets, *indices;

  cudaStream_t stream { nullptr };

  ALLOC_TRY((void**) &offsets, sizeof(int32_t) * (num_verts + 1), stream);
  ALLOC_TRY((void**) &indices, sizeof(int32_t) * num_edges, stream);

  gdf_column_view(result->adjList->offsets,
                  offsets,
                  nullptr,
                  num_verts + 1,
                  GDF_INT32);
  gdf_column_view(result->adjList->indices,
                  indices,
                  nullptr,
                  num_edges,
                  GDF_INT32);

  // Call nvgraphGetGraphStructure again to copy out the data
  topo.source_offsets = (int*)result->adjList->offsets->data;
  topo.destination_indices = (int*)result->adjList->indices->data;
  NVG_TRY(nvgraphGetGraphStructure(nvg_handle, nvg_result, (void*)&topo, &TT));

  
}

void triangle_count_nvgraph(Graph* G, uint64_t* result) {
  
  CUGRAPH_EXPECTS(G != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(G->adjList != nullptr, "Invalid API parameter");

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvg_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  createGraph_nvgraph(nvg_handle, G, &nvg_G, false);

  // Make Nvgraph call
  NVG_TRY(nvgraphTriangleCount(nvg_handle, nvg_G, result));
  
}

void louvain(Graph *graph, void *final_modularity, void *num_level, gdf_column *louvain_parts) {

  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter");

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

  void* louvain_parts_ptr = louvain_parts->data;

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
                 final_modularity, louvain_parts_ptr, num_level);
  
}

} //namespace cugraph
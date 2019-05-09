 /*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <cugraph.h>
#include <nvgraph_gdf.h>
#include <nvgraph/nvgraph.h>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"
#include <rmm_utils.h>

template<typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

gdf_error gdf_balancedCutClustering_nvgraph(gdf_graph* gdf_G,
                                            const int num_clusters,
                                            const int num_eigen_vects,
                                            const float evs_tolerance,
                                            const int evs_max_iter,
                                            const float kmean_tolerance,
                                            const int kmean_max_iter,
                                            gdf_column* clustering) {
  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((gdf_G->adjList != nullptr) || (gdf_G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!clustering->valid, GDF_VALIDITY_UNSUPPORTED);

  // Ensure that the input graph has values
  GDF_TRY(gdf_add_adj_list(gdf_G));

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
        cudaDataType_t settype;
        Vector<double> d_val;

  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false));
  int weight_index = 0;

        cudaStream_t stream { nullptr };
        rmm_temp_allocator allocator(stream);
        if (gdf_G->adjList->edge_data == nullptr) {
                // use a fp64 vector  [1,...,1]
                settype = CUDA_R_64F;
                d_val.resize(gdf_G->adjList->indices->size);
                thrust::fill(thrust::cuda::par(allocator).on(stream), d_val.begin(), d_val.end(), 1.0);
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
                                return GDF_UNSUPPORTED_DTYPE;
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
  return GDF_SUCCESS;
}

gdf_error gdf_spectralModularityMaximization_nvgraph(gdf_graph* gdf_G,
                                                      const int n_clusters,
                                                      const int n_eig_vects,
                                                      const float evs_tolerance,
                                                      const int evs_max_iter,
                                                      const float kmean_tolerance,
                                                      const int kmean_max_iter,
                                                      gdf_column* clustering) {
  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((gdf_G->adjList != nullptr) || (gdf_G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!clustering->valid, GDF_VALIDITY_UNSUPPORTED);

  // Ensure that the input graph has values
  GDF_TRY(gdf_add_adj_list(gdf_G));
  GDF_REQUIRE(gdf_G->adjList->edge_data != nullptr, GDF_INVALID_API_CALL);

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false));
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
  return GDF_SUCCESS;
}

gdf_error gdf_AnalyzeClustering_modularity_nvgraph(gdf_graph* gdf_G,
                                                    const int n_clusters,
                                                    gdf_column* clustering,
                                                    float* score) {
  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((gdf_G->adjList != nullptr) || (gdf_G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!clustering->valid, GDF_VALIDITY_UNSUPPORTED);

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false));
  int weight_index = 0;

  // Make Nvgraph call

  NVG_TRY(nvgraphAnalyzeClustering(nvg_handle,
                                    nvgraph_G,
                                    weight_index,
                                    n_clusters,
                                    (const int* )clustering->data,
                                    NVGRAPH_MODULARITY,
                                    score));
  return GDF_SUCCESS;
}

gdf_error gdf_AnalyzeClustering_edge_cut_nvgraph(gdf_graph* gdf_G,
                                                  const int n_clusters,
                                                  gdf_column* clustering,
                                                  float* score) {
  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((gdf_G->adjList != nullptr) || (gdf_G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!clustering->valid, GDF_VALIDITY_UNSUPPORTED);

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
        cudaDataType_t settype;
        Vector<double> d_val;

  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false));
  int weight_index = 0;

        cudaStream_t stream { nullptr };
        rmm_temp_allocator allocator(stream);
        if (gdf_G->adjList->edge_data == nullptr) {
                // use a fp64 vector  [1,...,1]
                settype = CUDA_R_64F;
                d_val.resize(gdf_G->adjList->indices->size);
                thrust::fill(thrust::cuda::par(allocator).on(stream), d_val.begin(), d_val.end(), 1.0);
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
                                return GDF_UNSUPPORTED_DTYPE;
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
  return GDF_SUCCESS;
}

gdf_error gdf_AnalyzeClustering_ratio_cut_nvgraph(gdf_graph* gdf_G,
                                                  const int n_clusters,
                                                  gdf_column* clustering,
                                                  float* score) {
  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((gdf_G->adjList != nullptr) || (gdf_G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(clustering->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!clustering->valid, GDF_VALIDITY_UNSUPPORTED);

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvgraph_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, false));
  int weight_index = 0;

  // Make Nvgraph call

  NVG_TRY(nvgraphAnalyzeClustering(nvg_handle,
                                    nvgraph_G,
                                    weight_index,
                                    n_clusters,
                                    (const int* )clustering->data,
                                    NVGRAPH_RATIO_CUT,
                                    score));
  return GDF_SUCCESS;
}


gdf_error gdf_extract_subgraph_vertex_nvgraph(gdf_graph* gdf_G,
                                              gdf_column* vertices,
                                              gdf_graph* result) {
  GDF_REQUIRE(gdf_G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((gdf_G->adjList != nullptr) || (gdf_G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE(vertices != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(vertices->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!vertices->valid, GDF_VALIDITY_UNSUPPORTED);

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvg_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvg_G, false));

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
    return GDF_C_ERROR;
  int num_verts = topo.nvertices;
  int num_edges = topo.nedges;
  result->adjList = new gdf_adj_list;
  result->adjList->offsets = new gdf_column;
  result->adjList->indices = new gdf_column;
  result->adjList->ownership = 0;
  int *offsets, *indices;

  cudaStream_t stream { nullptr };

  ALLOC_MANAGED_TRY((void**) &offsets, sizeof(int32_t) * (num_verts + 1), stream);
  ALLOC_MANAGED_TRY((void**) &indices, sizeof(int32_t) * num_edges, stream);

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

  return GDF_SUCCESS;
}

gdf_error gdf_triangle_count_nvgraph(gdf_graph* G, uint64_t* result) {
  GDF_REQUIRE(G != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE((G->adjList != nullptr) || (G->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_TRY(gdf_add_adj_list(G));
  GDF_REQUIRE(G->adjList != nullptr, GDF_INVALID_API_CALL);

  // Initialize Nvgraph and wrap the graph
  nvgraphHandle_t nvg_handle = nullptr;
  nvgraphGraphDescr_t nvg_G = nullptr;
  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, G, &nvg_G, false));

  // Make Nvgraph call
  NVG_TRY(nvgraphTriangleCount(nvg_handle, nvg_G, result));
  return GDF_SUCCESS;
}

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
 * @file nvgraph_gdf.h
 * ---------------------------------------------------------------------------**/

#pragma once

#include <cugraph.h>

/**
 * Takes a GDF graph and wraps its data with an Nvgraph graph object.
 * @param nvg_handle The Nvgraph handle
 * @param gdf_G Pointer to GDF graph object
 * @param nvgraph_G Pointer to the Nvgraph graph descriptor
 * @param use_transposed True if we are transposing the input graph while wrapping
 * @return Error code
 */
//gdf_error gdf_createGraph_nvgraph(nvgraphHandle_t nvg_handle,
//                                  gdf_graph* gdf_G,
//                                  nvgraphGraphDescr_t * nvgraph_G,
//                                  bool use_transposed = false);

/**
 * Wrapper function for Nvgraph SSSP algorithm
 * @param gdf_G Pointer to GDF graph object
 * @param source_vert Value for the starting vertex
 * @param sssp_distances Pointer to a GDF column in which the resulting distances will be stored
 * @return Error code
 */
gdf_error gdf_sssp_nvgraph(gdf_graph *gdf_G, const int *source_vert, gdf_column *sssp_distances);

/**
 * Wrapper function for Nvgraph balanced cut clustering
 * @param gdf_G Pointer to GDF graph object
 * @param num_clusters The desired number of clusters
 * @param num_eigen_vects The number of eigenvectors to use
 * @param evs_type The type of the eigenvalue solver to use
 * @param evs_tolerance The tolerance to use for the eigenvalue solver
 * @param evs_max_iter The maximum number of iterations of the eigenvalue solver
 * @param kmean_tolerance The tolerance to use for the kmeans solver
 * @param kmean_max_iter The maximum number of iteration of the k-means solver
 * @param clustering Pointer to a GDF column in which the resulting clustering will be stored
 * @param eig_vals Pointer to a GDF column in which the resulting eigenvalues will be stored
 * @param eig_vects Pointer to a GDF column in which the resulting eigenvectors will be stored
 * @return Error code
 */
gdf_error gdf_balancedCutClustering_nvgraph(gdf_graph* gdf_G,
					    const int num_clusters,
					    const int num_eigen_vects,
					    const float evs_tolerance,
					    const int evs_max_iter,
					    const float kmean_tolerance,
					    const int kmean_max_iter,
					    gdf_column* clustering);

/**
 * Wrapper function for Nvgraph spectral modularity maximization algorithm
 * @param gdf_G Pointer to GDF graph object
 * @param n_clusters The desired number of clusters
 * @param n_eig_vects The number of eigenvectors to use
 * @param evs_tolerance The tolerance to use for the eigenvalue solver
 * @param evs_max_iter The maximum number of iterations of the eigenvalue solver
 * @param kmean_tolerance The tolerance to use for the k-means solver
 * @param kmean_max_iter The maximum number of iterations of the k-means solver
 * @param clustering Pointer to a GDF column in which the resulting clustering will be stored
 * @param eig_vals Pointer to a GDF column in which the resulting eigenvalues will be stored
 * @param eig_vects Pointer to a GDF column in which the resulting eigenvectors will be stored
 * @return
 */
gdf_error gdf_spectralModularityMaximization_nvgraph(gdf_graph* gdf_G,
						     const int n_clusters,
						     const int n_eig_vects,
						     const float evs_tolerance,
						     const int evs_max_iter,
						     const float kmean_tolerance,
						     const int kmean_max_iter,
						     gdf_column* clustering);

/**
 * Wrapper function for Nvgraph clustering modularity metric
 * @param gdf_G Pointer to GDF graph object
 * @param n_clusters Number of clusters in the clustering
 * @param clustering Pointer to GDF column containing the clustering to analyze
 * @param score Pointer to a float in which the result will be written
 * @return Error code
 */
gdf_error gdf_AnalyzeClustering_modularity_nvgraph(gdf_graph* gdf_G,
                                                   const int n_clusters,
                                                   gdf_column* clustering,
                                                   float* score);

/**
 * Wrapper function for Nvgraph clustering edge cut metric
 * @param gdf_G Pointer to GDF graph object
 * @param n_clusters Number of clusters in the clustering
 * @param clustering Pointer to GDF column containing the clustering to analyze
 * @param score Pointer to a float in which the result will be written
 * @return Error code
 */
gdf_error gdf_AnalyzeClustering_edge_cut_nvgraph(gdf_graph* gdf_G,
						 const int n_clusters,
						 gdf_column* clustering,
						 float* score);

/**
 * Wrapper function for Nvgraph clustering ratio cut metric
 * @param gdf_G Pointer to GDF graph object
 * @param n_clusters Number of clusters in the clustering
 * @param clustering Pointer to GDF column containing the clustering to analyze
 * @param score Pointer to a float in which the result will be written
 * @return Error code
 */
gdf_error gdf_AnalyzeClustering_ratio_cut_nvgraph(gdf_graph* gdf_G,
						  const int n_clusters,
						  gdf_column* clustering,
						  float* score);

/**
 * Wrapper function for Nvgraph extract subgraph by vertices
 * @param gdf_G Pointer to GDF graph object, this is the input graph
 * @param vertices Pointer to GDF column object which contains the list of vertices to extract
 * @param result Pointer to GDF graph object, this is the output must be a valid pointer
 * @return Error code
 */
gdf_error gdf_extract_subgraph_vertex_nvgraph(gdf_graph* gdf_G,
                                              gdf_column* vertices,
                                              gdf_graph* result);
/**
 * Wrapper function for Nvgraph triangle counting
 * @param G Pointer to GDF graph object
 * @param result Pointer to a uint64_t in which the result will be written
 * @return Error code
 */
gdf_error gdf_triangle_count_nvgraph(gdf_graph* G, uint64_t* result);

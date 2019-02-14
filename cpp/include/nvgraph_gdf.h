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

#include <nvgraph/nvgraph.h>
#include <cugraph.h>

/**
 * Takes a GDF graph and wraps its data with an Nvgraph graph object.
 * @param nvg_handle The Nvgraph handle
 * @param gdf_G Pointer to GDF graph object
 * @param nvgraph_G Pointer to the Nvgraph graph descriptor
 * @param use_transposed True if we are transposing the input graph while wrapping
 * @return Error code
 */
gdf_error gdf_createGraph_nvgraph(nvgraphHandle_t nvg_handle,
																	gdf_graph* gdf_G,
																	nvgraphGraphDescr_t * nvgraph_G,
																	bool use_transposed = false);

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
 * @param num_eigen_vects The number of eigen-vectors to use
 * @param evs_type The type of the eigen-value solver to use
 * @param evs_tolerance The tolerance to use for the eigen-value solver
 * @param evs_max_iter The maximum number of iterations of the eigen-value solver
 * @param kmean_tolerance The tolerance to use for the kmeans solver
 * @param kmean_max_iter The maximum number of iteration of the kmeans solver
 * @param clustering Pointer to a GDF column in which the resulting clustering will be stored
 * @param eig_vals Pointer to a GDF column in which the resulting eigenvalues will be stored
 * @param eig_vects Pointer to a GDF column in which the resulting eigenvectors will be stored
 * @return Error code
 */
gdf_error gdf_balancedCutClustering_nvgraph(gdf_graph* gdf_G,
																						const int num_clusters,
																						const int num_eigen_vects,
																						const int evs_type,
																						const float evs_tolerance,
																						const int evs_max_iter,
																						const float kmean_tolerance,
																						const int kmean_max_iter,
																						gdf_column* clustering,
																						gdf_column* eig_vals,
																						gdf_column* eig_vects);

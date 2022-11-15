/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>
#include <cugraph_c/resource_handle.h>

/** @defgroup community Community algorithms
 *  @ingroup c_api
 *  @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque triangle counting result type
 */
typedef struct {
  int32_t align_;
} cugraph_triangle_count_result_t;

/**
 * @brief     Triangle Counting
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start        Device array of vertices we want to count triangles for.  If NULL
 *                           the entire set of vertices in the graph is processed
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result       Output from the triangle_count call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_triangle_count(const cugraph_resource_handle_t* handle,
                                            cugraph_graph_t* graph,
                                            const cugraph_type_erased_device_array_view_t* start,
                                            bool_t do_expensive_check,
                                            cugraph_triangle_count_result_t** result,
                                            cugraph_error_t** error);

/**
 * @brief     Get triangle counting vertices
 */
cugraph_type_erased_device_array_view_t* cugraph_triangle_count_result_get_vertices(
  cugraph_triangle_count_result_t* result);

/**
 * @brief     Get triangle counting counts
 */
cugraph_type_erased_device_array_view_t* cugraph_triangle_count_result_get_counts(
  cugraph_triangle_count_result_t* result);

/**
 * @brief     Free a triangle count result
 *
 * @param [in] result     The result from a sampling algorithm
 */
void cugraph_triangle_count_result_free(cugraph_triangle_count_result_t* result);

/**
 * @brief     Opaque heirarchical clustering output
 */
typedef struct {
  int32_t align_;
} cugraph_heirarchical_clustering_result_t;

/**
 * @brief     Compute Louvain
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  max_level    Maximum level in hierarchy
 * @param [in]  resolution   Resolution parameter (gamma) in modularity formula.
 *                           This changes the size of the communities.  Higher resolutions
 *                           lead to more smaller communities, lower resolutions lead to
 *                           fewer larger communities.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result       Output from the Louvain call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_louvain(const cugraph_resource_handle_t* handle,
                                     cugraph_graph_t* graph,
                                     size_t max_level,
                                     double resolution,
                                     bool_t do_expensive_check,
                                     cugraph_heirarchical_clustering_result_t** result,
                                     cugraph_error_t** error);

/**
 * @brief     Get heirarchical clustering vertices
 */
cugraph_type_erased_device_array_view_t* cugraph_heirarchical_clustering_result_get_vertices(
  cugraph_heirarchical_clustering_result_t* result);

/**
 * @brief     Get heirarchical clustering clusters
 */
cugraph_type_erased_device_array_view_t* cugraph_heirarchical_clustering_result_get_clusters(
  cugraph_heirarchical_clustering_result_t* result);

/**
 * @brief     Get modularity
 */
double cugraph_heirarchical_clustering_result_get_modularity(
  cugraph_heirarchical_clustering_result_t* result);

/**
 * @brief     Free a heirarchical clustering result
 *
 * @param [in] result     The result from a sampling algorithm
 */
void cugraph_heirarchical_clustering_result_free(cugraph_heirarchical_clustering_result_t* result);

/**
 * @brief   Extract ego graphs
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  source_vertices Device array of vertices we want to extract egonets for.
 * @param [in]  radius          The number of hops to go out from each source vertex
 * @param [in]  do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result           Opaque object containing the extracted subgraph
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_extract_ego(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  size_t radius,
  bool_t do_expensive_check,
  cugraph_induced_subgraph_result_t** result,
  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

/**
 *  @}
 */

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cugraph_c/random.h>
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
 * @brief     Opaque hierarchical clustering output
 */
typedef struct {
  int32_t align_;
} cugraph_hierarchical_clustering_result_t;

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
                                     cugraph_hierarchical_clustering_result_t** result,
                                     cugraph_error_t** error);

/**
 * @brief     Compute Leiden
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in/out] rng_state State of the random number generator, updated with each call
 * @param [in]  max_level    Maximum level in hierarchy
 * @param [in]  resolution   Resolution parameter (gamma) in modularity formula.
 *                           This changes the size of the communities.  Higher resolutions
 *                           lead to more smaller communities, lower resolutions lead to
 *                           fewer larger communities.
 * @param[in]  theta         (optional) The value of the parameter to scale modularity
 *                           gain in Leiden refinement phase. It is used to compute
 *                           the probability of joining a random leiden community.
 *                           Called theta in the Leiden algorithm.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result       Output from the Leiden call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_leiden(const cugraph_resource_handle_t* handle,
                                    cugraph_rng_state_t* rng_state,
                                    cugraph_graph_t* graph,
                                    size_t max_level,
                                    double resolution,
                                    double theta,
                                    bool_t do_expensive_check,
                                    cugraph_hierarchical_clustering_result_t** result,
                                    cugraph_error_t** error);

/**
 * @brief     Get hierarchical clustering vertices
 */
cugraph_type_erased_device_array_view_t* cugraph_hierarchical_clustering_result_get_vertices(
  cugraph_hierarchical_clustering_result_t* result);

/**
 * @brief     Get hierarchical clustering clusters
 */
cugraph_type_erased_device_array_view_t* cugraph_hierarchical_clustering_result_get_clusters(
  cugraph_hierarchical_clustering_result_t* result);

/**
 * @brief     Get modularity
 */
double cugraph_hierarchical_clustering_result_get_modularity(
  cugraph_hierarchical_clustering_result_t* result);

/**
 * @brief     Free a hierarchical clustering result
 *
 * @param [in] result     The result from a sampling algorithm
 */
void cugraph_hierarchical_clustering_result_free(cugraph_hierarchical_clustering_result_t* result);

/**
 * @brief     Compute ECG clustering of the given graph
 *
 * ECG runs truncated Louvain on an ensemble of permutations of the input graph,
 * then uses the ensemble partitions to determine weights for the input graph.
 * The final result is found by running full Louvain on the input graph using
 * the determined weights. See https://arxiv.org/abs/1809.05578 for further
 * information.
 *
 * NOTE: This currently wraps the legacy ECG clustering implementation which is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 * @param [in]  min_weight      The minimum weight parameter
 * @param [in]  ensemble_size   The ensemble size parameter
 * @param [in]  do_expensive_check
 *                              A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result          The result from the clustering algorithm
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_ecg(const cugraph_resource_handle_t* handle,
                                 cugraph_graph_t* graph,
                                 double min_weight,
                                 size_t ensemble_size,
                                 bool_t do_expensive_check,
                                 cugraph_hierarchical_clustering_result_t** result,
                                 cugraph_error_t** error);

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

/**
 * @brief     Opaque clustering output
 */
typedef struct {
  int32_t align_;
} cugraph_clustering_result_t;

/**
 * @brief   Balanced cut clustering
 *
 * NOTE: This currently wraps the legacy balanced cut clustering implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  n_eigenvectors  The number of eigenvectors to use
 * @param [in]  evs_tolerance   The tolerance to use for the eigenvalue solver
 * @param [in]  evs_max_iterations The maximum number of iterations of the eigenvalue solver
 * @param [in]  k_means_tolerance  The tolerance to use for the k-means solver
 * @param [in]  k_means_max_iterations The maximum number of iterations of the k-means solver
 * @param [in]  do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result           Opaque object containing the clustering result
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_balanced_cut_clustering(const cugraph_resource_handle_t* handle,
                                                     cugraph_graph_t* graph,
                                                     size_t n_clusters,
                                                     size_t n_eigenvectors,
                                                     double evs_tolerance,
                                                     int evs_max_iterations,
                                                     double k_means_tolerance,
                                                     int k_means_max_iterations,
                                                     bool_t do_expensive_check,
                                                     cugraph_clustering_result_t** result,
                                                     cugraph_error_t** error);

/**
 * @brief   Spectral clustering
 *
 * NOTE: This currently wraps the legacy spectral clustering implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  n_eigenvectors  The number of eigenvectors to use
 * @param [in]  evs_tolerance   The tolerance to use for the eigenvalue solver
 * @param [in]  evs_max_iterations The maximum number of iterations of the eigenvalue solver
 * @param [in]  k_means_tolerance  The tolerance to use for the k-means solver
 * @param [in]  k_means_max_iterations The maximum number of iterations of the k-means solver
 * @param [in]  do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result           Opaque object containing the clustering result
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_spectral_modularity_maximization(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  size_t n_eigenvectors,
  double evs_tolerance,
  int evs_max_iterations,
  double k_means_tolerance,
  int k_means_max_iterations,
  bool_t do_expensive_check,
  cugraph_clustering_result_t** result,
  cugraph_error_t** error);

/**
 * @brief   Compute modularity of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral modularity implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  vertices        Vertex ids from the clustering result
 * @param [in]  clusters        Cluster ids from the clustering result
 * @param [out] score           The modularity score for this clustering
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_analyze_clustering_modularity(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* clusters,
  double* score,
  cugraph_error_t** error);

/**
 * @brief   Compute edge cut of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral edge cut implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  vertices        Vertex ids from the clustering result
 * @param [in]  clusters        Cluster ids from the clustering result
 * @param [out] score           The edge cut score for this clustering
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_analyze_clustering_edge_cut(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* clusters,
  double* score,
  cugraph_error_t** error);

/**
 * @brief   Compute ratio cut of the specified clustering
 *
 * NOTE: This currently wraps the legacy spectral ratio cut implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  n_clusters      The desired number of clusters
 * @param [in]  vertices        Vertex ids from the clustering result
 * @param [in]  clusters        Cluster ids from the clustering result
 * @param [out] score           The ratio cut score for this clustering
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_analyze_clustering_ratio_cut(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  size_t n_clusters,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* clusters,
  double* score,
  cugraph_error_t** error);

/**
 * @brief     Get clustering vertices
 */
cugraph_type_erased_device_array_view_t* cugraph_clustering_result_get_vertices(
  cugraph_clustering_result_t* result);

/**
 * @brief     Get clustering clusters
 */
cugraph_type_erased_device_array_view_t* cugraph_clustering_result_get_clusters(
  cugraph_clustering_result_t* result);

/**
 * @brief     Free a clustering result
 *
 * @param [in] result     The result from a sampling algorithm
 */
void cugraph_clustering_result_free(cugraph_clustering_result_t* result);

#ifdef __cplusplus
}
#endif

/**
 *  @}
 */

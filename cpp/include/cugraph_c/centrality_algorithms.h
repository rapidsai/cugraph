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

#include <cugraph_c/array.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

/** @defgroup centrality Centrality algorithms
 *  @ingroup c_api
 *  @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque centrality result type
 */
typedef struct {
  int32_t align_;
} cugraph_centrality_result_t;

/**
 * @brief     Get the vertex ids from the centrality result
 *
 * @param [in]   result   The result from a centrality algorithm
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_centrality_result_get_vertices(
  cugraph_centrality_result_t* result);

/**
 * @brief     Get the centrality values from a centrality algorithm result
 *
 * @param [in]   result   The result from a centrality algorithm
 * @return type erased array view of centrality values
 */
cugraph_type_erased_device_array_view_t* cugraph_centrality_result_get_values(
  cugraph_centrality_result_t* result);

/**
 * @brief     Free centrality result
 *
 * @param [in]   result   The result from a centrality algorithm
 */
void cugraph_centrality_result_free(cugraph_centrality_result_t* result);

/**
 * @brief     Compute pagerank
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sume of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  has_initial_guess If set to `true`, values in the PageRank output array (pointed by
 * @p pageranks) is used as initial PageRank values. If false, initial PageRank values are set
 * to 1.0 divided by the number of vertices in the graph.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t has_initial_guess,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Compute personalized pagerank
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  precomputed_vertex_out_weight_sums
 *                          Optionally send in precomputed sume of vertex out weights
 *                          (a performance optimization).  Set to NULL if
 *                          no value is passed.
 * FIXME:  Make this just [in], copy it if I need to temporarily modify internally
 * @param [in/out]  personalization_vertices Pointer to an array storing personalization vertex
 * identifiers (compute personalized PageRank).  Array might be modified if renumbering is enabled
 * for the graph
 * @param [in]  personalization_values Pointer to an array storing personalization values for the
 * vertices in the personalization set.
 * @param [in]  alpha       PageRank damping factor.
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in PageRank values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations Maximum number of PageRank iterations.
 * @param [in]  has_initial_guess If set to `true`, values in the PageRank output array (pointed by
 * @p pageranks) is used as initial PageRank values. If false, initial PageRank values are set
 * to 1.0 divided by the number of vertices in the graph.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_personalized_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
  // FIXME:  Make this const, copy it if I need to temporarily modify internally
  cugraph_type_erased_device_array_view_t* personalization_vertices,
  const cugraph_type_erased_device_array_view_t* personalization_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t has_initial_guess,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Compute eigenvector centrality
 *
 * Computed using the power method.
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is measured
 *                          comparing the L1 norm until it is less than epsilon
 * @param [in]  max_iterations Maximum number of power iterations, will not exceed this number
 *                          of iterations even if we haven't converged
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to eigenvector centrality results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_eigenvector_centrality(const cugraph_resource_handle_t* handle,
                                                    cugraph_graph_t* graph,
                                                    double epsilon,
                                                    size_t max_iterations,
                                                    bool_t do_expensive_check,
                                                    cugraph_centrality_result_t** result,
                                                    cugraph_error_t** error);

/**
 * @brief     Compute katz centrality
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  betas       Optionally send in a device array holding values to be added to
 *                          each vertex's new Katz Centrality score in every iteration.
 *                          If set to NULL then @p beta is used for all vertices.
 * @param [in]  alpha       Katz centrality attenuation factor.  This should be smaller
 *                          than the inverse of the maximum eigenvalue of this graph
 * @param [in]  beta        Constant value to be added to each vertex's new Katz
 *                          Centrality score in every iteration.  Relevant only when
 *                          @p betas is NULL
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in Katz Centrality values between
 *                          two consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon. (L1-norm)
 * @param [in]  max_iterations Maximum number of Katz Centrality iterations.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to katz centrality results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_katz_centrality(const cugraph_resource_handle_t* handle,
                                             cugraph_graph_t* graph,
                                             const cugraph_type_erased_device_array_view_t* betas,
                                             double alpha,
                                             double beta,
                                             double epsilon,
                                             size_t max_iterations,
                                             bool_t do_expensive_check,
                                             cugraph_centrality_result_t** result,
                                             cugraph_error_t** error);

/**
 * @brief     Opaque hits result type
 */
typedef struct {
  int32_t align_;
} cugraph_hits_result_t;

/**
 * @brief     Get the vertex ids from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_hits_result_get_vertices(
  cugraph_hits_result_t* result);

/**
 * @brief     Get the hubs values from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of hubs values
 */
cugraph_type_erased_device_array_view_t* cugraph_hits_result_get_hubs(
  cugraph_hits_result_t* result);

/**
 * @brief     Get the authorities values from the hits result
 *
 * @param [in]   result   The result from hits
 * @return type erased array of authorities values
 */
cugraph_type_erased_device_array_view_t* cugraph_hits_result_get_authorities(
  cugraph_hits_result_t* result);

/**
 * @brief   Get the score differences between the last two iterations
 *
 * @param [in]   result   The result from hits
 * @return score differences
 */
double cugraph_hits_result_get_hub_score_differences(cugraph_hits_result_t* result);

/**
 * @brief   Get the actual number of iterations
 *
 * @param [in]   result   The result from hits
 * @return actual number of iterations
 */
size_t cugraph_hits_result_get_number_of_iterations(cugraph_hits_result_t* result);

/**
 * @brief     Free hits result
 *
 * @param [in]   result   The result from hits
 */
void cugraph_hits_result_free(cugraph_hits_result_t* result);

/**
 * @brief     Compute hits
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  epsilon     Error tolerance to check convergence. Convergence is assumed
 *                          if the sum of the differences in Hits values between two
 *                          consecutive iterations is less than the number of vertices
 *                          in the graph multiplied by @p epsilon.
 * @param [in]  max_iterations
 *                          Maximum number of Hits iterations.
 * @param [in]  initial_hubs_guess_vertices
 *                          Pointer to optional type erased device array containing
 *                          the vertex ids for an initial hubs guess.  If set to NULL
 *                          there is no initial guess.
 * @param [in]  initial_hubs_guess_values
 *                          Pointer to optional type erased device array containing
 *                          the values for an initial hubs guess.  If set to NULL
 *                          there is no initial guess.  Note that both
 *                          @p initial_hubs_guess_vertices and @p initial_hubs_guess_values
 *                          have to be specified (or they both have to be NULL).  Otherwise
 *                          this will be treated as an error.
 * @param [in]  normalize   A flag to normalize the results (if set to `true`)
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to hits results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_hits(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  double epsilon,
  size_t max_iterations,
  const cugraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
  const cugraph_type_erased_device_array_view_t* initial_hubs_guess_values,
  bool_t normalize,
  bool_t do_expensive_check,
  cugraph_hits_result_t** result,
  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

/**
 *  @}
 */

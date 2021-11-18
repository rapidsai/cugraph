/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/cugraph_api.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque pagerank result type
 */
typedef struct {
  int align_;
} cugraph_pagerank_result_t;

/**
 * @brief     Get the vertex ids from the pagerank result
 *
 * @param [in]   result   The result from pagerank
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_t* cugraph_pagerank_result_get_vertices(
  cugraph_pagerank_result_t* result);

/**
 * @brief     Get the pagerank values from the pagerank result
 *
 * @param [in]   result   The result from pagerank
 * @return type erased array of pagerank values
 */
cugraph_type_erased_device_array_t* cugraph_pagerank_result_get_pageranks(
  cugraph_pagerank_result_t* result);

/**
 * @brief     Free pagerank result
 *
 * @param [in]   result   The result from pagerank
 */
void cugraph_pagerank_result_free(cugraph_pagerank_result_t* result);

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
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t has_initial_guess,
  bool_t do_expensive_check,
  cugraph_pagerank_result_t** result,
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
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @param [out] result      Opaque pointer to pagerank results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_personalized_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
  cugraph_type_erased_device_array_t* personalization_vertices,
  const cugraph_type_erased_device_array_t* personalization_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t has_initial_guess,
  bool_t do_expensive_check,
  cugraph_pagerank_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Opaque bfs result type
 */
typedef struct {
  int align_;
} cugraph_bfs_result_t;

/**
 * @brief     Get the vertex ids from the bfs result
 *
 * @param [in]   result   The result from bfs
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_t* cugraph_bfs_result_get_vertices(cugraph_bfs_result_t* result);

/**
 * @brief     Get the distances from the bfs result
 *
 * @param [in]   result   The result from bfs
 * @return type erased array of distances
 */
cugraph_type_erased_device_array_t* cugraph_bfs_result_get_distances(cugraph_bfs_result_t* result);

/**
 * @brief     Get the predecessors from the bfs result
 *
 * @param [in]   result   The result from bfs
 * @return type erased array of predecessors.  Value will be NULL if
 *         compute_predecessors was FALSE in the call to bfs that
 *         produced this result.
 */
cugraph_type_erased_device_array_t* cugraph_bfs_result_get_predecessors(
  cugraph_bfs_result_t* result);

/**
 * @brief     Free bfs result
 *
 * @param [in]   result   The result from bfs
 */
void cugraph_bfs_result_free(cugraph_bfs_result_t* result);

/**
 * @brief     Perform a breadth first search from a set of seed vertices.
 *
 * This function computes the distances (minimum number of hops to reach the vertex) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in/out]  sources  Array of source vertices.  NOTE: Array might be modified if
 *                           renumbering is enabled for the graph
 * @param [in]  direction_optimizing If set to true, this algorithm switches between the push based
 * breadth-first search and pull based breadth-first search depending on the size of the
 * breadth-first search frontier (currently unsupported). This option is valid only for symmetric
 * input graphs.
 * @param depth_limit Sets the maximum number of breadth-first search iterations. Any vertices
 * farther than @p depth_limit hops from @p source_vertex will be marked as unreachable.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @param [out] result       Opaque pointer to pagerank results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_bfs(const cugraph_resource_handle_t* handle,
                                 cugraph_graph_t* graph,
                                 cugraph_type_erased_device_array_t* sources,
                                 bool_t direction_optimizing,
                                 size_t depth_limit,
                                 bool_t do_expensive_check,
                                 bool_t compute_predecessors,
                                 cugraph_bfs_result_t** result,
                                 cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

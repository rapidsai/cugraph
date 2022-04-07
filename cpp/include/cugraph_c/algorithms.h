/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cugraph_c/resource_handle.h>

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

/**
 * @brief     Opaque paths result type
 *
 * Store the output of BFS or SSSP, computing predecessors and distances
 * from a seed.
 */
typedef struct {
  int32_t align_;
} cugraph_paths_result_t;

/**
 * @brief     Get the vertex ids from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_vertices(
  cugraph_paths_result_t* result);

/**
 * @brief     Get the distances from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of distances
 */
cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_distances(
  cugraph_paths_result_t* result);

/**
 * @brief     Get the predecessors from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of predecessors.  Value will be NULL if
 *         compute_predecessors was FALSE in the call to bfs or sssp that
 *         produced this result.
 */
cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_predecessors(
  cugraph_paths_result_t* result);

/**
 * @brief     Free paths result
 *
 * @param [in]   result   The result from bfs or sssp
 */
void cugraph_paths_result_free(cugraph_paths_result_t* result);

/**
 * @brief     Perform a breadth first search from a set of seed vertices.
 *
 * This function computes the distances (minimum number of hops to reach the vertex) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * FIXME:  Make this just [in], copy it if I need to temporarily modify internally
 * @param [in/out]  sources  Array of source vertices.  NOTE: Array might be modified if
 *                           renumbering is enabled for the graph
 * @param [in]  direction_optimizing If set to true, this algorithm switches between the push based
 * breadth-first search and pull based breadth-first search depending on the size of the
 * breadth-first search frontier (currently unsupported). This option is valid only for symmetric
 * input graphs.
 * @param depth_limit Sets the maximum number of breadth-first search iterations. Any vertices
 * farther than @p depth_limit hops from @p source_vertex will be marked as unreachable.
 * @param [in] compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in] do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_bfs(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  // FIXME:  Make this const, copy it if I need to temporarily modify internally
  cugraph_type_erased_device_array_view_t* sources,
  bool_t direction_optimizing,
  size_t depth_limit,
  bool_t compute_predecessors,
  bool_t do_expensive_check,
  cugraph_paths_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Perform single-source shortest-path to compute the minimum distances
 *            (and predecessors) from the source vertex.
 *
 * This function computes the distances (minimum edge weight sums) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  source       Source vertex id
 * @param [in]  cutoff       Maximum edge weight sum to consider
 * @param [in]  compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_sssp(const cugraph_resource_handle_t* handle,
                                  cugraph_graph_t* graph,
                                  size_t source,
                                  double cutoff,
                                  bool_t compute_predecessors,
                                  bool_t do_expensive_check,
                                  cugraph_paths_result_t** result,
                                  cugraph_error_t** error);

/**
 * @brief     Opaque extract_paths result type
 */
typedef struct {
  int32_t align_;
} cugraph_extract_paths_result_t;

/**
 * @brief     Extract BFS or SSSP paths from a cugraph_paths_result_t
 *
 * This function extracts paths from the BFS or SSSP output.  BFS and SSSP output
 * distances and predecessors.  The path from a vertex v back to the original
 * source vertex can be extracted by recursively looking up the predecessor
 * vertex until you arrive back at the original source vertex.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  result       Output from the BFS call
 * @param [in]  destinations Array of destination vertices.
 * @param [out] result       Opaque pointer to extract_paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_extract_paths(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* sources,
  const cugraph_paths_result_t* paths_result,
  const cugraph_type_erased_device_array_view_t* destinations,
  cugraph_extract_paths_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Get the max path length from extract_paths result
 *
 * @param [in]   result   The result from extract_paths
 * @return maximum path length
 */
size_t cugraph_extract_paths_result_get_max_path_length(cugraph_extract_paths_result_t* result);

/**
 * @brief     Get the matrix (row major order) of paths
 *
 * @param [in]   result   The result from extract_paths
 * @return type erased array pointing to the matrix in device memory
 */
cugraph_type_erased_device_array_view_t* cugraph_extract_paths_result_get_paths(
  cugraph_extract_paths_result_t* result);

/**
 * @brief     Free extract_paths result
 *
 * @param [in]   result   The result from extract_paths
 */
void cugraph_extract_paths_result_free(cugraph_extract_paths_result_t* result);

/**
 * @brief     Opaque random walk result type
 */
typedef struct {
  int32_t align_;
} cugraph_random_walk_result_t;

/**
 * @brief  Compute random walks using the node2vec framework.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  max_depth    Maximum length of the generated path
 * @param [in]  compress_result If true, return the paths as a compressed sparse row matrix,
 *                              otherwise return as a dense matrix
 * @param [in]  p            The return parameter
 * @param [in]  q            The in/out parameter
 * @param [in]  result       Output from the node2vec call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_node2vec(const cugraph_resource_handle_t* handle,
                                      cugraph_graph_t* graph,
                                      const cugraph_type_erased_device_array_view_t* sources,
                                      size_t max_depth,
                                      bool_t compress_result,
                                      double p,
                                      double q,
                                      cugraph_random_walk_result_t** result,
                                      cugraph_error_t** error);

/**
 * @brief     Get the max path length from random walk result
 *
 * @param [in]   result   The result from random walks
 * @return maximum path length
 */
size_t cugraph_random_walk_result_get_max_path_length(cugraph_random_walk_result_t* result);

// FIXME:  Should this be the same as extract_paths_result_t?  The only
//         difference at the moment is that RW results contain weights
//         and extract_paths results don't.  But that's probably wrong.
/**
 * @brief     Get the matrix (row major order) of vertices in the paths
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path matrix in device memory
 */
cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_paths(
  cugraph_random_walk_result_t* result);

/**
 * @brief     Get the matrix (row major order) of edge weights in the paths
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path edge weights in device memory
 */
cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_weights(
  cugraph_random_walk_result_t* result);

/**
 * @brief     If the random walk result is compressed, get the path sizes
 *
 * @param [in]   result   The result from a random walk algorithm
 * @return type erased array pointing to the path sizes in device memory
 */
cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_path_sizes(
  cugraph_random_walk_result_t* result);

/**
 * @brief     Free random walks result
 *
 * @param [in]   result   The result from random walks
 */
void cugraph_random_walk_result_free(cugraph_random_walk_result_t* result);

/**
 * @brief     Opaque neighborhood sampling result type
 */
typedef struct {
  int32_t align_;
} cugraph_sample_result_t;

/**
 * @brief     Uniform Neighborhood Sampling
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start        Device array of start vertices for the sampling
 * @param [in]  start_label  Device array of start labels.  These labels will propagate to the
 * results so that the result can be properly organized when the input needs to be sent back to
 * different callers (different processes or different gpus).
 * @param [in]  fanout       Host array defining the fan out at each step in the sampling algorithm
 * @param [in]  with_replacement
 *                           Boolean value.  If true selection of edges is done with
 *                           replacement.  If false selection is done without replacement.
 * @param [in]  do_expensive_check
 *                           A flag to run expensive checks for input arguments (if set to true)
 * @param [in]  result       Output from the uniform_neighbor_sample call
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_uniform_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start,
  const cugraph_type_erased_device_array_view_t* start_label,
  const cugraph_type_erased_host_array_view_t* fan_out,
  bool_t with_replacement,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Get the source vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the source vertices in device memory
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_sources(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the destination vertices from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the destination vertices in device memory
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_destinations(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the start labels from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the start labels
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_start_labels(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the index from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the index
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_index(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the transaction counts from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased host array pointing to the counts
 */
cugraph_type_erased_host_array_view_t* cugraph_sample_result_get_counts(
  const cugraph_sample_result_t* result);

/**
 * @brief     Free a sampling result
 *
 * @param [in]   result   The result from a sampling algorithm
 */
void cugraph_sample_result_free(cugraph_sample_result_t* result);

#ifdef __cplusplus
}
#endif

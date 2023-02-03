/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <cugraph_c/random.h>
#include <cugraph_c/resource_handle.h>

/** @defgroup sampling Sampling algorithms
 *  @ingroup c_api
 *  @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque random walk result type
 */
typedef struct {
  int32_t align_;
} cugraph_random_walk_result_t;

/**
 * @brief  Compute uniform random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_uniform_random_walks(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  cugraph_random_walk_result_t** result,
  cugraph_error_t** error);

/**
 * @brief  Compute biased random walks
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_biased_random_walks(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  cugraph_random_walk_result_t** result,
  cugraph_error_t** error);

/**
 * @brief  Compute random walks using the node2vec framework.
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                              needs to be transposed
 * @param [in]  start_vertices  Array of source vertices
 * @param [in]  max_length      Maximum length of the generated path
 * @param [in]  compress_result If true, return the paths as a compressed sparse row matrix,
 *                              otherwise return as a dense matrix
 * @param [in]  p               The return parameter
 * @param [in]  q               The in/out parameter
 * @param [in]  result          Output from the node2vec call
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_node2vec_random_walks(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  double p,
  double q,
  cugraph_random_walk_result_t** result,
  cugraph_error_t** error);

/**
 * @brief  Compute random walks using the node2vec framework.
 * @deprecated This call should be replaced with cugraph_node2vec_random_walks
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
 * @deprecated This call will no longer be relevant once the new node2vec are called
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
 * @deprecated This call should be replaced with cugraph_uniform_neighborhood_sampling
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start        Device array of start vertices for the sampling
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
  const cugraph_type_erased_host_array_view_t* fan_out,
  bool_t with_replacement,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Uniform Neighborhood Sampling
 *
 * Returns a sample of the neighborhood around specified start vertices.  Optionally, each
 * start vertex can be associated with a label, allowing the caller to specify multiple batches
 * of sampling requests in the same function call - which should improve GPU utilization.
 *
 * If label is NULL then all start vertices will be considered part of the same batch and the
 * return value will not have a label column.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  start        Device array of start vertices for the sampling
 * @param [in]  label        Device array of start labels for the sampling.  The labels associated
 * with each start vertex will be included in the output associated with results that were derived
 * from that start vertex.  We only support label of type INT32. If label is NULL, the return data
 * will not be labeled.
 * @param [in]  fanout       Host array defining the fan out at each step in the sampling algorithm.
 *                           We only support fanout values of type INT32
 * @param [in/out] rng_state State of the random number generator, updated with each call
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
cugraph_error_code_t cugraph_uniform_neighbor_sample_with_edge_properties(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start,
  const cugraph_type_erased_device_array_view_t* label,
  const cugraph_type_erased_host_array_view_t* fan_out,
  cugraph_rng_state_t* rng_state,
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
 * @brief     Get the edge_id from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_id
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_id(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the edge_type from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_type
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_type(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the edge_weight from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the edge_weight
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_weight(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the hop from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the hop
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_hop(
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
// FIXME:  This will be obsolete when the older mechanism is removed
cugraph_type_erased_host_array_view_t* cugraph_sample_result_get_counts(
  const cugraph_sample_result_t* result);

/**
 * @brief     Free a sampling result
 *
 * @param [in]   result   The result from a sampling algorithm
 */
void cugraph_sample_result_free(cugraph_sample_result_t* result);

/**
 * @brief     Create a sampling result (testing API)
 *
 * @param [in]   handle         Handle for accessing resources
 * @param [in]   srcs           Device array view to populate srcs
 * @param [in]   dsts           Device array view to populate dsts
 * @param [in]   edge_id        Device array view to populate edge_id (can be NULL)
 * @param [in]   edge_type      Device array view to populate edge_type (can be NULL)
 * @param [in]   wgt            Device array view to populate wgt (can be NULL)
 * @param [in]   hop            Device array view to populate hop
 * @param [in]   label          Device array view to populate label (can be NULL)
 * @param [out]  result         Pointer to the location to store the
 *                              cugraph_sample_result_t*
 * @param [out]  error          Pointer to an error object storing details of
 *                              any error.  Will be populated if error code is
 *                              not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_test_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* edge_id,
  const cugraph_type_erased_device_array_view_t* edge_type,
  const cugraph_type_erased_device_array_view_t* wgt,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error);

/**
 * @brief     Create a sampling result (testing API)
 *
 * @param [in]   handle         Handle for accessing resources
 * @param [in]   srcs           Device array view to populate srcs
 * @param [in]   dsts           Device array view to populate dsts
 * @param [in]   edge_id        Device array view to populate edge_id
 * @param [in]   edge_type      Device array view to populate edge_type
 * @param [in]   weight         Device array view to populate weight
 * @param [in]   hop            Device array view to populate hop
 * @param [in]   label          Device array view to populate label
 * @param [out]  result         Pointer to the location to store the
 *                              cugraph_sample_result_t*
 * @param [out]  error          Pointer to an error object storing details of
 *                              any error.  Will be populated if error code is
 *                              not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_test_uniform_neighborhood_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* edge_id,
  const cugraph_type_erased_device_array_view_t* edge_type,
  const cugraph_type_erased_device_array_view_t* weight,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

/**
 *  @}
 */

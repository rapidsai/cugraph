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
 * @brief     Opaque sampling options type
 */
typedef struct {
  int32_t align_;
} cugraph_sampling_options_t;

/**
 * @brief     Enumeration for prior sources behavior
 */
typedef enum cugraph_prior_sources_behavior_t {
  DEFAULT = 0, /** Construct sources for hop k from destination vertices from hop k-1 */
  CARRY_OVER,  /** Construct sources for hop k from destination vertices from hop k-1
                   and sources from hop k-1 */
  EXCLUDE      /** Construct sources for hop k from destination vertices form hop k-1,
                   but exclude any vertex that has already been used as a source */
} cugraph_prior_sources_behavior_t;

/**
 * @brief   Create sampling options object
 *
 * All sampling options set to FALSE
 *
 * @param [out] options Opaque pointer to the sampling options
 * @param [out] error   Pointer to an error object storing details of any error.  Will
 *                      be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_sampling_options_create(cugraph_sampling_options_t** options,
                                                     cugraph_error_t** error);

/**
 * @brief   Set flag to renumber results
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
void cugraph_sampling_set_renumber_results(cugraph_sampling_options_t* options, bool_t value);

/**
 * @brief   Set flag to sample with_replacement
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
void cugraph_sampling_set_with_replacement(cugraph_sampling_options_t* options, bool_t value);

/**
 * @brief   Set flag to sample return_hops
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
void cugraph_sampling_set_return_hops(cugraph_sampling_options_t* options, bool_t value);

/**
 * @brief   Set prior sources behavior
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Enum defining prior sources behavior
 */
void cugraph_sampling_set_prior_sources_behavior(cugraph_sampling_options_t* options,
                                                 cugraph_prior_sources_behavior_t value);

/**
 * @brief   Set flag to sample dedupe_sources prior to sampling
 *
 * @param options - opaque pointer to the sampling options
 * @param value - Boolean value to assign to the option
 */
void cugraph_sampling_set_dedupe_sources(cugraph_sampling_options_t* options, bool_t value);

/**
 * @brief     Free sampling options object
 *
 * @param [in]   options   Opaque pointer to sampling object
 */
void cugraph_sampling_options_free(cugraph_sampling_options_t* options);

/**
 * @brief     Uniform Neighborhood Sampling
 * @deprecated This call should be replaced with cugraph_uniform_neighbor_sample
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
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  start_vertex_labels  Device array of start vertex labels for the sampling.  The
 * labels associated with each start vertex will be included in the output associated with results
 * that were derived from that start vertex.  We only support label of type INT32. If label is
 * NULL, the return data will not be labeled.
 * @param [in]  label_list Device array of the labels included in @p start_vertex_labels.  If
 * @p label_to_comm_rank is not specified this parameter is ignored.  If specified, label_list
 * must be sorted in ascending order.
 * @param [in]  label_to_comm_rank Device array identifying which comm rank the output for a
 * particular label should be shuffled in the output.  If not specifed the data is not organized in
 * output.  If specified then the all data from @p label_list[i] will be shuffled to rank @p
 * label_to_comm_rank[i].  If not specified then the output data will not be shuffled between ranks.
 * @param [in]  fanout       Host array defining the fan out at each step in the sampling algorithm.
 *                           We only support fanout values of type INT32
 * @param [in/out] rng_state State of the random number generator, updated with each call
 * @param [in]  with_replacement
 *                           Boolean value.  If true selection of edges is done with
 *                           replacement.  If false selection is done without replacement.
 * @param [in]  return_hops  Boolean value.  If true include the hop number in the result,
 *                           If false the hop number will not be included in result.
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
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* start_vertex_labels,
  const cugraph_type_erased_device_array_view_t* label_list,
  const cugraph_type_erased_device_array_view_t* label_to_comm_rank,
  const cugraph_type_erased_host_array_view_t* fan_out,
  cugraph_rng_state_t* rng_state,
  bool_t with_replacement,
  bool_t return_hops,
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
 * @param [in]  start_vertices Device array of start vertices for the sampling
 * @param [in]  start_vertex_labels  Device array of start vertex labels for the sampling.  The
 * labels associated with each start vertex will be included in the output associated with results
 * that were derived from that start vertex.  We only support label of type INT32. If label is
 * NULL, the return data will not be labeled.
 * @param [in]  label_list Device array of the labels included in @p start_vertex_labels.  If
 * @p label_to_comm_rank is not specified this parameter is ignored.  If specified, label_list
 * must be sorted in ascending order.
 * @param [in]  label_to_comm_rank Device array identifying which comm rank the output for a
 * particular label should be shuffled in the output.  If not specifed the data is not organized in
 * output.  If specified then the all data from @p label_list[i] will be shuffled to rank @p.  This
 * cannot be specified unless @p start_vertex_labels is also specified
 * label_to_comm_rank[i].  If not specified then the output data will not be shuffled between ranks.
 * @param [in]  fanout       Host array defining the fan out at each step in the sampling algorithm.
 *                           We only support fanout values of type INT32
 * @param [in/out] rng_state State of the random number generator, updated with each call
 * @param [in]  sampling_options
 *                           Opaque pointer defining the sampling options.
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
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* start_vertex_labels,
  const cugraph_type_erased_device_array_view_t* label_list,
  const cugraph_type_erased_device_array_view_t* label_to_comm_rank,
  const cugraph_type_erased_host_array_view_t* fan_out,
  cugraph_rng_state_t* rng_state,
  const cugraph_sampling_options_t* options,
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
 * @brief     Get the result offsets from the sampling algorithm result
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the result offsets
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_offsets(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the renumber map
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_renumber_map(
  const cugraph_sample_result_t* result);

/**
 * @brief     Get the renumber map offsets
 *
 * @param [in]   result   The result from a sampling algorithm
 * @return type erased array pointing to the renumber map offsets
 */
cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_renumber_map_offsets(
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

/**
 * @brief Select random vertices from the graph
 *
 * @param [in]      handle        Handle for accessing resources
 * @param [in]      graph         Pointer to graph
 * @param [in/out]  rng_state     State of the random number generator, updated with each call
 * @param [in]      num_vertices  Number of vertices to sample
 * @param [out]     vertices      Device array view to populate label
 * @param [out]     error         Pointer to an error object storing details of
 *                                any error.  Will be populated if error code is
 *                                not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_select_random_vertices(const cugraph_resource_handle_t* handle,
                                                    const cugraph_graph_t* graph,
                                                    cugraph_rng_state_t* rng_state,
                                                    size_t num_vertices,
                                                    cugraph_type_erased_device_array_t** vertices,
                                                    cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

/**
 *  @}
 */

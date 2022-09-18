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
#include <cugraph_c/graph_functions.h>
#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque similarity result type
 */
typedef struct {
  int32_t align_;
} cugraph_similarity_result_t;

/**
 * @brief       Get the similarity coefficient array
 *
 * @param [in]     result   The result from a similarity algorithm
 * @return type erased array of similarity coefficients
 */
cugraph_type_erased_device_array_view_t* cugraph_similarity_result_get_similarity(
  cugraph_similarity_result_t* result);

/**
 * @brief     Free similarity result
 *
 * @param [in]    result    The result from a similarity algorithm
 */
void cugraph_similarity_result_free(cugraph_similarity_result_t* result);

/**
 * @brief     Perform Jaccard similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that Jaccard similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_jaccard_coefficients(const cugraph_resource_handle_t* handle,
                                                  cugraph_graph_t* graph,
                                                  const cugraph_vertex_pairs_t* vertex_pairs,
                                                  bool_t use_weight,
                                                  bool_t do_expensive_check,
                                                  cugraph_similarity_result_t** result,
                                                  cugraph_error_t** error);

/**
 * @brief     Perform Sorensen similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that Sorensen similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_sorensen_coefficients(const cugraph_resource_handle_t* handle,
                                                   cugraph_graph_t* graph,
                                                   const cugraph_vertex_pairs_t* vertex_pairs,
                                                   bool_t use_weight,
                                                   bool_t do_expensive_check,
                                                   cugraph_similarity_result_t** result,
                                                   cugraph_error_t** error);

/**
 * @brief     Perform overlap similarity computation
 *
 * Compute the similarity for the specified vertex_pairs
 *
 * Note that overlap similarity must run on a symmetric graph.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  vertex_pairs Vertex pair for input
 * @param [in]  use_weight   If true consider the edge weight in the graph, if false use an
 *                           edge weight of 1
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to similarity results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_overlap_coefficients(const cugraph_resource_handle_t* handle,
                                                  cugraph_graph_t* graph,
                                                  const cugraph_vertex_pairs_t* vertex_pairs,
                                                  bool_t use_weight,
                                                  bool_t do_expensive_check,
                                                  cugraph_similarity_result_t** result,
                                                  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

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
#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup labeling Labeling algorithms
 *  @ingroup c_api
 *  @{
 */

/**
 * @brief     Opaque labeling result type
 */
typedef struct {
  int32_t align_;
} cugraph_labeling_result_t;

/**
 * @brief     Get the vertex ids from the labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_labeling_result_get_vertices(
  cugraph_labeling_result_t* result);

/**
 * @brief     Get the label values from the labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 * @return type erased array of label values
 */
cugraph_type_erased_device_array_view_t* cugraph_labeling_result_get_labels(
  cugraph_labeling_result_t* result);

/**
 * @brief     Free labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 */
void cugraph_labeling_result_free(cugraph_labeling_result_t* result);

/**
 * @brief Labels each vertex in the input graph with its (weakly-connected-)component ID
 *
 * The input graph must be symmetric. Component IDs can be arbitrary integers (they can be
 * non-consecutive and are not ordered by component size or any other criterion).
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to labeling results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_weakly_connected_components(const cugraph_resource_handle_t* handle,
                                                         cugraph_graph_t* graph,
                                                         bool_t do_expensive_check,
                                                         cugraph_labeling_result_t** result,
                                                         cugraph_error_t** error);

/**
 * @brief Labels each vertex in the input graph with its (strongly-connected-)component ID
 *
 * The input graph may be asymmetric. Component IDs can be arbitrary integers (they can be
 * non-consecutive and are not ordered by component size or any other criterion).
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to labeling results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_strongly_connected_components(const cugraph_resource_handle_t* handle,
                                                           cugraph_graph_t* graph,
                                                           bool_t do_expensive_check,
                                                           cugraph_labeling_result_t** result,
                                                           cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

/**
 *  @}
 */

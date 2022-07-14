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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief       Opaque core number result type
 */
typedef struct {
  int32_t align_;
} cugraph_core_result_t;

/**
 * @brief       Get the vertex ids from the core result
 *
 * @param [in]     result   The result from core number
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_core_result_get_vertices(
  cugraph_core_result_t* result);

/**
 * @brief       Get the core numbers from the core result
 *
 * @param [in]    result    The result from core number
 * @return type erased array of core numbers
 */
cugraph_type_erased_device_array_view_t* cugraph_core_result_get_core_numbers(
  cugraph_core_result_t* result);

/**
 * @brief     Free core result
 *
 * @param [in]    result    The result from core number
 */
void cugraph_core_result_free(cugraph_core_result_t* result);

/**
 * @brief     Enumeration for computing core number
 */
typedef enum {
  K_CORE_DEGREE_TYPE_IN    = 0, /** Compute core_number using incoming edges */
  K_CORE_DEGREE_TYPE_OUT   = 1, /** Compute core_number using outgoing edges */
  K_CORE_DEGREE_TYPE_INOUT = 2  /** Compute core_number using both incoming and outgoing edges */
} cugraph_k_core_degree_type_t;

/**
 * @brief     Perform core number.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  degree_type  Compute core_number using in, out or both in and out edges
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_core_number(const cugraph_resource_handle_t* handle,
                                         cugraph_graph_t* graph,
                                         cugraph_k_core_degree_type_t degree_type,
                                         bool_t do_expensive_check,
                                         cugraph_core_result_t** result,
                                         cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

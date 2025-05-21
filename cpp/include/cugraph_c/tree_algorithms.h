/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

/** @defgroup layout Layout algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque layout output
 */
typedef struct {
  int32_t align_;
} cugraph_layout_result_t;

/**
 * @brief   Minimum Spanning Tree
 *
 * NOTE: This currently wraps the legacy minimum implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]   handle          Handle for accessing resources
 * @param [in]   graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                               needs to be transposed
 * @param [in]   do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result          Opaque object containing the extracted subgraph
 * @param [out]  error           Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_minimum_spanning_tree(const cugraph_resource_handle_t* handle,
                                                   cugraph_graph_t* graph,
                                                   bool_t do_expensive_check,
                                                   cugraph_induced_subgraph_result_t** result,
                                                   cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

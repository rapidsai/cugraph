/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/array.h>
#include <cugraph_c/error.h>
#include <cugraph_c/export.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup components Component algorithms
 */

/**
 * @brief     Opaque labeling result type
 */
typedef struct {
  int32_t align_;
} cugraph_labeling_result_t;

/**
 * @ingroup components
 * @brief     Get the vertex ids from the labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 * @return type erased array of vertex ids
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_labeling_result_get_vertices(
  cugraph_labeling_result_t* result);

/**
 * @ingroup components
 * @brief     Get the label values from the labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 * @return type erased array of label values
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_labeling_result_get_labels(
  cugraph_labeling_result_t* result);

/**
 * @ingroup components
 * @brief     Free labeling result
 *
 * @param [in]   result   The result from a labeling algorithm
 */
CUGRAPH_EXPORT void cugraph_labeling_result_free(cugraph_labeling_result_t* result);

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
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_weakly_connected_components(const cugraph_resource_handle_t* handle,
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
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_strongly_connected_components(const cugraph_resource_handle_t* handle,
                                      cugraph_graph_t* graph,
                                      bool_t do_expensive_check,
                                      cugraph_labeling_result_t** result,
                                      cugraph_error_t** error);

/**
 * @brief     Opaque simple cycles result type
 */
typedef struct {
  int32_t align_;
} cugraph_simple_cycles_result_t;

/**
 * @ingroup components
 * @brief     Get the flat concatenation of cycle vertex sequences
 *
 * Vertices of cycle @p i occupy
 * [cycle_offsets[i], cycle_offsets[i + 1]) in this array, listed in cyclic order.
 * Vertex ids are in user/external id space.
 *
 * @param [in]   result   The result from cugraph_simple_cycles
 * @return type erased array of cycle vertex ids
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t*
cugraph_simple_cycles_result_get_cycle_vertices(cugraph_simple_cycles_result_t* result);

/**
 * @ingroup components
 * @brief     Get CSR-style offsets into the cycle vertex array
 *
 * Device array of type SIZE_T with length (number of cycles on this GPU) + 1.
 *
 * @param [in]   result   The result from cugraph_simple_cycles
 * @return type erased array of cycle offsets
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t*
cugraph_simple_cycles_result_get_cycle_offsets(cugraph_simple_cycles_result_t* result);

/**
 * @ingroup components
 * @brief     Get the number of simple cycles stored on this GPU
 *
 * @param [in]   result   The result from cugraph_simple_cycles
 * @return number of cycles on this GPU
 */
CUGRAPH_EXPORT size_t
cugraph_simple_cycles_result_get_num_cycles(cugraph_simple_cycles_result_t* result);

/**
 * @ingroup components
 * @brief     Free simple cycles result
 *
 * @param [in]   result   The result from cugraph_simple_cycles
 */
CUGRAPH_EXPORT void cugraph_simple_cycles_result_free(cugraph_simple_cycles_result_t* result);

/**
 * @brief Enumerate simple cycles (elementary circuits) in a directed graph
 *
 * The algorithm is defined for directed (asymmetric) graphs. Symmetric graphs are not supported
 * and are not rejected at the API boundary; results may be incorrect.
 *
 * This function is intended for small to moderate @p length_bound values (for example, no larger
 * than 10).
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph. NOTE: Graph storage may be transposed if required.
 * @param [in]  seed_vertices Optional device array of seed vertex ids in user/external id space.
 *                          Order does not matter. If NULL, all simple cycles with length <=
 *                          @p length_bound are returned. If non-NULL, only cycles that contain at
 *                          least one seed are returned. Multi-GPU: if any rank uses seed filtering,
 *                          every rank must pass a non-NULL array; ranks with no local seeds pass
 *                          an empty array (size 0), not NULL.
 * @param [in]  length_bound Maximum cycle length (must be > 0).
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result      Opaque pointer to simple cycles results
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 */
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_simple_cycles(const cugraph_resource_handle_t* handle,
                      cugraph_graph_t* graph,
                      const cugraph_type_erased_device_array_view_t* seed_vertices,
                      size_t length_bound,
                      bool_t do_expensive_check,
                      cugraph_simple_cycles_result_t** result,
                      cugraph_error_t** error);

#include <cugraph_c/export.h>

#ifdef __cplusplus
}
#endif

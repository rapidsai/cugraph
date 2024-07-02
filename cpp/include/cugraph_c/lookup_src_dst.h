/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque src-dst lookup container type
 */

typedef struct {
  int32_t align_;
} cugraph_lookup_container_t;

/**
 * @brief Opaque src-dst lookup result type
 */

typedef struct {
  int32_t align_;
} cugraph_lookup_result_t;

/**
 * @brief Build map to lookup source and destination using edge id and type
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [out]  lookup_container Lookup map
 * @param [out]  error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_build_edge_id_and_type_to_src_dst_lookup_map(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph_lookup_container_t** lookup_container,
  cugraph_error_t** error);

/**
 * @brief Lookup edge sources and destinations using edge ids and a single edge type.
 *
 * Use this function to lookup endpoints of edges belonging to the same edge type.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  lookup_container Lookup map
 * @param[in]  edge_ids_to_lookup Edge ids to lookup
 * @param[in]  edge_type_to_lookup Edge types corresponding to edge ids in @p edge_ids_to_lookup
 * @param [out]  result      Output from the lookup call
 * @param [out]  error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_single_type(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  int edge_type_to_lookup,
  cugraph_lookup_result_t** result,
  cugraph_error_t** error);

/**
 * @brief Lookup edge sources and destinations using edge ids and edge types.
 *
 * Use this function to lookup endpoints of edges belonging to different edge types.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  lookup_container Lookup map
 * @param[in]  edge_ids_to_lookup Edge ids to lookup
 * @param[in]  edge_types_to_lookup Edge types corresponding to the edge ids in @p
 * edge_ids_to_lookup
 * @param [out]  result      Output from the lookup call
 * @param [out]  error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_types(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  const cugraph_type_erased_device_array_view_t* edge_types_to_lookup,
  cugraph_lookup_result_t** result,
  cugraph_error_t** error);

/**
 * @ingroup samplingC
 * @brief  Get the edge sources from the lookup result
 *
 * @param [in]  result  The result from src-dst lookup using edge ids and type(s)
 * @return type erased array pointing to the edge sources
 */
cugraph_type_erased_device_array_view_t* cugraph_lookup_result_get_srcs(
  const cugraph_lookup_result_t* result);

/**
 * @ingroup samplingC
 * @brief  Get the edge destinations from the lookup result
 *
 * @param [in]  result  The result from src-dst lookup using edge ids and type(s)
 * @return type erased array pointing to the edge destinations
 */
cugraph_type_erased_device_array_view_t* cugraph_lookup_result_get_dsts(
  const cugraph_lookup_result_t* result);

/**
 * @ingroup samplingC
 * @brief     Free a src-dst lookup result
 *
 * @param [in] result     The result from src-dst lookup using edge ids and type(s)
 */
void cugraph_lookup_result_free(cugraph_lookup_result_t* result);

#ifdef __cplusplus
}
#endif

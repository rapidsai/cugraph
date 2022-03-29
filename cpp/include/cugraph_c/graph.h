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

#include <cugraph_c/array.h>
#include <cugraph_c/cugraph_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} cugraph_graph_t;

typedef struct {
  bool_t is_symmetric;
  bool_t is_multigraph;
} cugraph_graph_properties_t;

// FIXME: Add support for specifying isolated vertices
/**
 * @brief     Construct an SG graph
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  properties  Properties of the constructed graph
 * @param [in]  src         Device array containing the source vertex ids.
 * @param [in]  dst         Device array containing the destination vertex ids
 * @param [in]  weights     Device array containing the edge weights
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber    If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering is required if the vertices are not sequential
 *    integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [in]  properties  Properties of the graph
 * @param [out] graph       A pointer to the graph object
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_sg_graph_create(const cugraph_resource_handle_t* handle,
                                             const cugraph_graph_properties_t* properties,
                                             const cugraph_type_erased_device_array_view_t* src,
                                             const cugraph_type_erased_device_array_view_t* dst,
                                             const cugraph_type_erased_device_array_view_t* weights,
                                             bool_t store_transposed,
                                             bool_t renumber,
                                             bool_t check,
                                             cugraph_graph_t** graph,
                                             cugraph_error_t** error);

/**
 * @brief     Destroy an SG graph
 *
 * @param [in]  graph  A pointer to the graph object to destroy
 */
// FIXME:  This should probably just be cugraph_graph_free
//         but didn't want to confuse with original cugraph_free_graph
void cugraph_sg_graph_free(cugraph_graph_t* graph);

// FIXME: Add support for specifying isolated vertices
/**
 * @brief     Construct an MG graph
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  properties  Properties of the constructed graph
 * @param [in]  src         Device array containing the source vertex ids
 * @param [in]  dst         Device array containing the destination vertex ids
 * @param [in]  weights     Device array containing the edge weights
 * @param [in]  vertex_partition_offsets Host array containing the offsets for each vertex
 * partition
 * @param [in]  segment_offsets Host array containing the offsets for each segment
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  num_vertices  Number of vertices
 * @param [in]  num_edges   Number of edges
 * @param [in]  check       If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph       A pointer to the graph object
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_mg_graph_create(const cugraph_resource_handle_t* handle,
                                             const cugraph_graph_properties_t* properties,
                                             const cugraph_type_erased_device_array_view_t* src,
                                             const cugraph_type_erased_device_array_view_t* dst,
                                             const cugraph_type_erased_device_array_view_t* weights,
                                             bool_t store_transposed,
                                             size_t num_edges,
                                             bool_t check,
                                             cugraph_graph_t** graph,
                                             cugraph_error_t** error);

/**
 * @brief     Destroy an MG graph
 *
 * @param [in]  graph  A pointer to the graph object to destroy
 */
// FIXME:  This should probably just be cugraph_graph_free
//         but didn't want to confuse with original cugraph_free_graph
void cugraph_mg_graph_free(cugraph_graph_t* graph);

#ifdef __cplusplus
}
#endif

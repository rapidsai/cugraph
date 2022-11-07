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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} cugraph_graph_t;

typedef struct {
  int32_t align_;
} cugraph_data_mask_t;

typedef struct {
  bool_t is_symmetric;
  bool_t is_multigraph;
} cugraph_graph_properties_t;

// FIXME: Add support for specifying isolated vertices
/**
 * @brief     Construct an SG graph
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  properties     Properties of the constructed graph
 * @param [in]  src            Device array containing the source vertex ids.
 * @param [in]  dst            Device array containing the destination vertex ids
 * @param [in]  weights        Device array containing the edge weights.  Note that an unweighted
 *                             graph can be created by passing weights == NULL.
 * @param [in]  edge_ids       Device array containing the edge ids for each edge.  Optional
                               argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                               argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber       If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering is required if the vertices are not sequential
 *    integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [in]  properties     Properties of the graph
 * @param [out] graph          A pointer to the graph object
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_sg_graph_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_type_ids,
  bool_t store_transposed,
  bool_t renumber,
  bool_t check,
  cugraph_graph_t** graph,
  cugraph_error_t** error);

/**
 * @brief     Construct an SG graph from a CSR input
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  properties     Properties of the constructed graph
 * @param [in]  offsets        Device array containing the CSR offsets array
 * @param [in]  indices        Device array containing the destination vertex ids
 * @param [in]  weights        Device array containing the edge weights.  Note that an unweighted
 *                             graph can be created by passing weights == NULL.
 * @param [in]  edge_ids       Device array containing the edge ids for each edge.  Optional
                               argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                               argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  renumber       If true, renumber vertices to make an efficient data structure.
 *    If false, do not renumber.  Renumbering is required if the vertices are not sequential
 *    integer values from 0 to num_vertices.
 * @param [in]  do_expensive_check    If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [in]  properties     Properties of the graph
 * @param [out] graph          A pointer to the graph object
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_sg_graph_create_from_csr(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* offsets,
  const cugraph_type_erased_device_array_view_t* indices,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_type_ids,
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
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  properties      Properties of the constructed graph
 * @param [in]  src             Device array containing the source vertex ids
 * @param [in]  dst             Device array containing the destination vertex ids
 * @param [in]  weights         Device array containing the edge weights.  Note that an unweighted
 *                              graph can be created by passing weights == NULL.  If a weighted
 *                              graph is to be created, the weights device array should be created
 *                              on each rank, but the pointer can be NULL and the size 0
 *                              if there are no inputs provided by this rank
 * @param [in]  edge_ids        Device array containing the edge ids for each edge.  Optional
                                argument that can be NULL if edge ids are not used.
 * @param [in]  edge_type_ids  Device array containing the edge types for each edge.  Optional
                                argument that can be NULL if edge types are not used.
 * @param [in]  store_transposed If true create the graph initially in transposed format
 * @param [in]  num_edges       Number of edges
 * @param [in]  check           If true, do expensive checks to validate the input data
 *    is consistent with software assumptions.  If false bypass these checks.
 * @param [out] graph           A pointer to the graph object
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_mg_graph_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_device_array_view_t* edge_ids,
  const cugraph_type_erased_device_array_view_t* edge_type_ids,
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

/**
 * @brief     Create a data mask
 *
 * @param [in]  handle          Handle for accessing resources
 * @param [in]  vertex_bit_mask Device array containing vertex bit mask
 * @param [in]  edge_bit_mask   Device array containing edge bit mask
 * @param [in]  complement      If true, a 0 in one of the bit masks implies
 *                              the vertex/edge should be included and a 1 should
 *                              be excluded.  If false a 1 in one of the bit masks
 *                              implies the vertex/edge should be included and a 0
 *                              should be excluded.
 * @param [out] mask            An opaque pointer to the constructed mask object
 * @param [out] error           Pointer to an error object storing details of any error.  Will
 *                              be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_data_mask_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* vertex_bit_mask,
  const cugraph_type_erased_device_array_view_t* edge_bit_mask,
  bool_t complement,
  cugraph_data_mask_t** mask,
  cugraph_error_t** error);

/**
 * @brief     Get the data mask currently associated with a graph
 *
 * @param [in]  graph       The input graph
 * @param [out] mask        Opaque pointer where we should store the
 *                          current mask.  Will be NULL if there is no mask
 *                          currently assigned to the graph.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_graph_get_data_mask(cugraph_graph_t* graph,
                                                 cugraph_data_mask_t** mask,
                                                 cugraph_error_t** error);

/**
 * @brief     Associate a data mask with a graph
 *
 * NOTE: This function will fail if there is already a data mask associated with this graph
 *
 * @param [in]  graph       The input graph
 * @param [out] mask        Opaque pointer of the new data mask
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_graph_add_data_mask(cugraph_graph_t* graph,
                                                 cugraph_data_mask_t* mask,
                                                 cugraph_error_t** error);

/**
 * @brief     Release the data mask currently associated with a graph
 *
 * This function will remove the associated of the current data mask
 * with this graph.  The caller will be responsible for destroying the data
 * mask using graph_data_mask_destroy.
 *
 * If this function is not called and the graph is destroyed, the act of destroying
 * the graph will also destroy the data mask.
 *
 * If this function is called on a graph that is not currently associated with
 * a graph, then the mask will be set to NULL.
 *
 * @param [in]  graph       The input graph
 * @param [out] mask        Opaque pointer where we should store the
 *                          current mask.
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_graph_release_data_mask(cugraph_graph_t* graph,
                                                     cugraph_data_mask_t** mask,
                                                     cugraph_error_t** error);

/**
 * @brief     Destroy a data mask
 *
 * @param [in]  mask  A pointer to the data mask to destroy
 */
void cugraph_data_mask_destroy(cugraph_data_mask_t* mask);

#ifdef __cplusplus
}
#endif

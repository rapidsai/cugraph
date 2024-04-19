/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
 * @brief       Opaque vertex pair type
 */
typedef struct {
  int32_t align_;
} cugraph_vertex_pairs_t;

/**
 * @brief       Create vertex_pairs
 *
 * Input data will be shuffled to the proper GPU and stored in the
 * output vertex_pairs.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Graph to operate on
 * @param [in]  first        Type erased array of vertex ids for the first vertex of the pair
 * @param [in]  second       Type erased array of vertex ids for the second vertex of the pair
 * @param [out] vertex_pairs Opaque pointer to vertex_pairs
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_create_vertex_pairs(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* first,
  const cugraph_type_erased_device_array_view_t* second,
  cugraph_vertex_pairs_t** vertex_pairs,
  cugraph_error_t** error);

/**
 * @brief       Get the first vertex id array
 *
 * @param [in]     vertex_pairs   A vertex_pairs
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_vertex_pairs_get_first(
  cugraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief       Get the second vertex id array
 *
 * @param [in]     vertex_pairs   A vertex_pairs
 * @return type erased array of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_vertex_pairs_get_second(
  cugraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief     Free vertex pair
 *
 * @param [in]    vertex_pairs The vertex pairs
 */
void cugraph_vertex_pairs_free(cugraph_vertex_pairs_t* vertex_pairs);

/**
 * @brief      Find all 2-hop neighbors in the graph
 *
 * @param [in]  handle         Handle for accessing resources
 * @param [in]  graph          Pointer to graph
 * @param [in]  start_vertices Optional type erased array of starting vertices
 *                             If NULL use all, if specified compute two-hop
 *                             neighbors for these starting vertices
 * @param [in]  do_expensive_check
 *                             A flag to run expensive checks for input arguments (if set to true)
 * @param [out] result         Opaque pointer to resulting vertex pairs
 * @param [out] error          Pointer to an error object storing details of any error.  Will
 *                             be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_two_hop_neighbors(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  bool_t do_expensive_check,
  cugraph_vertex_pairs_t** result,
  cugraph_error_t** error);

/**
 * @brief       Opaque induced subgraph type
 */
typedef struct {
  int32_t align_;
} cugraph_induced_subgraph_result_t;

/**
 * @brief       Get the source vertex ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of source vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_sources(
  cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the destination vertex ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of destination vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_destinations(
  cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge weights
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge weights
 */
cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_edge_weights(
  cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge ids
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge ids
 */
cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_edge_ids(
  cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the edge types
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of edge types
 */
cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_edge_type_ids(
  cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief       Get the subgraph offsets
 *
 * @param [in]     induced_subgraph   Opaque pointer to induced subgraph
 * @return type erased array view of subgraph identifiers
 */
cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_subgraph_offsets(
  cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief     Free induced subgraph
 *
 * @param [in]    induced subgraph   Opaque pointer to induced subgraph
 */
void cugraph_induced_subgraph_result_free(cugraph_induced_subgraph_result_t* induced_subgraph);

/**
 * @brief      Extract induced subgraph(s)
 *
 * Given a list of vertex ids, extract a list of edges that represent the subgraph
 * containing only the specified vertex ids.
 *
 * This function will do multiple subgraph extractions concurrently.  The vertex ids
 * are specified in CSR-style, with @p subgraph_vertices being a list of vertex ids
 * and @p subgraph_offsets[i] identifying the start offset for each extracted subgraph
 *
 * @param [in]  handle            Handle for accessing resources
 * @param [in]  graph             Pointer to graph
 * @param [in]  subgraph_offsets  Type erased array of subgraph offsets into
 *                                @p subgraph_vertices
 * @param [in]  subgraph_vertices Type erased array of vertices to include in
 *                                extracted subgraph.
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result            Opaque pointer to induced subgraph result
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_extract_induced_subgraph(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* subgraph_offsets,
  const cugraph_type_erased_device_array_view_t* subgraph_vertices,
  bool_t do_expensive_check,
  cugraph_induced_subgraph_result_t** result,
  cugraph_error_t** error);

cugraph_error_code_t cugraph_lookup_src_dst_from_edge_id(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  bool_t do_expensive_check,
  cugraph_vertex_pairs_t** result,
  cugraph_error_t** error);

// FIXME: Rename the return type
/**
 * @brief      Gather edgelist
 *
 * This function collects the edgelist from all ranks and stores the combine edgelist
 * in each rank
 *
 * @param [in]  handle            Handle for accessing resources.
 * @param [in]  src               Device array containing the source vertex ids.
 * @param [in]  dst               Device array containing the destination vertex ids
 * @param [in]  weights           Optional device array containing the edge weights
 * @param [in]  edge_ids          Optional device array containing the edge ids for each edge.
 * @param [in]  edge_type_ids     Optional device array containing the edge types for each edge
 * @param [out] result            Opaque pointer to gathered edgelist result
 * @param [out] error             Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_allgather(const cugraph_resource_handle_t* handle,
                                       const cugraph_type_erased_device_array_view_t* src,
                                       const cugraph_type_erased_device_array_view_t* dst,
                                       const cugraph_type_erased_device_array_view_t* weights,
                                       const cugraph_type_erased_device_array_view_t* edge_ids,
                                       const cugraph_type_erased_device_array_view_t* edge_type_ids,
                                       cugraph_induced_subgraph_result_t** result,
                                       cugraph_error_t** error);

/**
 * @brief       Opaque degree result type
 */
typedef struct {
  int32_t align_;
} cugraph_degrees_result_t;

/**
 * @brief      Compute in degrees
 *
 * Compute the in degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute in degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_in_degrees(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  bool_t do_expensive_check,
  cugraph_degrees_result_t** result,
  cugraph_error_t** error);

/**
 * @brief      Compute out degrees
 *
 * Compute the out degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute out degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_out_degrees(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* source_vertices,
  bool_t do_expensive_check,
  cugraph_degrees_result_t** result,
  cugraph_error_t** error);

/**
 * @brief      Compute degrees
 *
 * Compute the degrees for the vertices in the graph.
 *
 * @param [in]  handle              Handle for accessing resources.
 * @param [in]  graph               Pointer to graph
 * @param [in]  source_vertices     Device array of vertices we want to compute degrees for.
 * @param [in]  do_expensive_check  A flag to run expensive checks for input arguments (if set to
 * true)
 * @param [out] result              Opaque pointer to degrees result
 * @param [out] error               Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_degrees(const cugraph_resource_handle_t* handle,
                                     cugraph_graph_t* graph,
                                     const cugraph_type_erased_device_array_view_t* source_vertices,
                                     bool_t do_expensive_check,
                                     cugraph_degrees_result_t** result,
                                     cugraph_error_t** error);

/**
 * @brief       Get the vertex ids
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_vertices(
  cugraph_degrees_result_t* degrees_result);

/**
 * @brief       Get the in degrees
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_in_degrees(
  cugraph_degrees_result_t* degrees_result);

/**
 * @brief       Get the out degrees
 *
 * If the graph is symmetric, in degrees and out degrees will be equal (and
 * will be stored in the same memory).
 *
 * @param [in]     degrees_result   Opaque pointer to degree result
 * @return type erased array view of vertex ids
 */
cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_out_degrees(
  cugraph_degrees_result_t* degrees_result);

/**
 * @brief     Free degree result
 *
 * @param [in]    degrees_result   Opaque pointer to degree result
 */
void cugraph_degrees_result_free(cugraph_degrees_result_t* degrees_result);

#ifdef __cplusplus
}
#endif

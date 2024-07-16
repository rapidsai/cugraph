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

//
// Speculative description of handling generic vertex and edge properties.
//
// If we have vertex properties and edge properties that we want to apply to an existing graph
// (after it was created) we could use these methods to construct C++ objects to represent these
// properties.
//
// These assume the use of external vertex ids and external edge ids as the mechanism for
// correlating a property to a particular vertex or edge.
//

#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} cugraph_vertex_property_t;

typedef struct {
  int32_t align_;
} cugraph_edge_property_t;

typedef struct {
  int32_t align_;
} cugraph_vertex_property_view_t;

typedef struct {
  int32_t align_;
} cugraph_edge_property_view_t;

#if 0
// Blocking out definition of these since this is speculative work.

/**
 * @brief     Create a vertex property
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  graph       Pointer to graph.
 * @param [in]  vertex_ids  Device array of vertex ids
 * @param [in]  property    Device array of vertex property
 * @param [out] result      Pointer to the location to store the pointer to the vertex property object
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_vertex_property_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_t * graph,
  const cugraph_type_erased_device_array_t* vertex_ids,
  const cugraph_type_erased_device_array_t* properties,
  cugraph_vertex_property_t** result,
  cugraph_error_t** error);

/**
 * @brief     Create a edge property
 *
 * @param [in]  handle           Handle for accessing resources
 * @param [in]  graph            Pointer to graph.
 * @param [in]  lookup_container Lookup map
 * @param [in]  edge_ids         Device array of edge ids
 * @param [in]  property         Device array of edge property
 * @param [out] result           Pointer to the location to store the pointer to the edge property object
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_edge_property_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_t * graph,
  const cugraph_lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_t* edge_ids,
  const cugraph_type_erased_device_array_t* properties,
  cugraph_edge_property_t** result,
  cugraph_error_t** error);

/**
 * @brief     Create a vertex_property_view from a vertex property
 *
 * @param [in]  vertex_property   Pointer to the vertex property object
 * @return Pointer to the view of the host array
 */
cugraph_vertex_property_view_t* cugraph_vertex_property_view(
  cugraph_vertex_property_view* vertex_property);

/**
 * @brief     Create a edge_property_view from a edge property
 *
 * @param [in]  edge_property   Pointer to the edge property object
 * @return Pointer to the view of the host array
 */
cugraph_edge_property_view_t* cugraph_edge_property_view(
  cugraph_edge_property_view* edge_property);

/**
 * @brief    Destroy a vertex_property object
 *
 * @param [in]  p    Pointer to the vertex_property object
 */
void cugraph_vertex_property_free(cugraph_vertex_property_t* p);

/**
 * @brief    Destroy a edge_property object
 *
 * @param [in]  p    Pointer to the edge_property object
 */
void cugraph_edge_property_free(cugraph_edge_property_t* p);

/**
 * @brief    Destroy a vertex_property_view object
 *
 * @param [in]  p    Pointer to the vertex_property_view object
 */
void cugraph_vertex_property_view_free(cugraph_vertex_property__viewt* p);

/**
 * @brief    Destroy a edge_property_view object
 *
 * @param [in]  p    Pointer to the edge_property_view object
 */
void cugraph_edge_property_view_free(cugraph_edge_property_view_t* p);
#endif

#ifdef __cplusplus
}
#endif

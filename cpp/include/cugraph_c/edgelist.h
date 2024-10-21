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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} cugraph_edgelist_t;

typedef struct {
  int32_t align_;
} cugraph_single_edgelist_view_t;

/**
 * @brief     Create an edgelist objects.  A graph can be constructed by a collection
 *            of lists of edges.  Logically these will be concatenated into a single list.
 *
 * @param [in]  handle           Handle for accessing resources
 * @param [in]  num_lists        Number of edgelist view to be populated
 * @param [in]  store_transposed If true the edges are specified in transposed format
 * @param [out] edgelists        Pointer to constructed edgelist object
 * @param [out] error            Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_create(const cugraph_resource_handle_t* handle,
                                             size_t num_lists,
                                             bool_t store_transposed,
                                             cugraph_edgelist_t** edgelist,
                                             cugraph_error_t** error);

/**
 * @brief    Set number of edges on an edgelist
 *
 * @param [in]  edgelist    Edge list to update
 * @param [in]  list_index  Index of edge list to update
 * @param [in]  num_edges   Number of edges
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 */
cugraph_error_code_t cugraph_edgelist_set_num_edges(cugraph_edgelist_t* edgelist,
                                                    size_t list_index,
                                                    size_t num_edges,
                                                    cugraph_error_t** error);

/**
 * @brief     Add vertices to edgelist object
 *
 * @param [in]     handle              Handle for accessing resources
 * @param [in/out] edgelist            A pointer to the edgelist object
 * @param [in]     list_index          Index of edgelist to update
 * @param [in]     num_vertices        Number of vertices specified.  In MG context this would
 *                                     be the number of vertices specified on this GPU
 * @param [in]     dtype               Datatype for vertices
 * @param [out]    vertices_array_view Pointer to view of vertices array
 * @param [out]    error               Pointer to an error object storing details of any error. Will
 *                                     be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_vertices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t num_vertices,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** vertices_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add major vertices to edgelist object.  For a non-transposed graph these are the
 *            source vertices of the edges.  For a transposed graph these are the destination
 *            vertices.
 *
 * Note that an edgelist should be created either with offsets/indices or with
 * majors/minors, specifying both is an error.
 *
 * @param [in]     handle            Handle for accessing resources
 * @param [in/out] edgelist          A pointer to the edgelist object
 * @param [in]     list_index        Index of edgelist to update
 * @param [in]     size              Number of entries in list
 * @param [in]     dtype             Datatype for major vertices
 * @param [out]    majors_array_view Pointer to view of majors array
 * @param [out]    error             Pointer to an error object storing details of any error.  Will
 *                                   be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_major_vertices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** majors_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add minor vertices to edgelist object.  For a non-transposed graph these are the
 *            destination vertices of the edges.  For a transposed graph these are the source
 *            vertices.
 *
 * Note that an edgelist should be created either with offsets/indices or with
 * majors/minors, specifying both is an error.
 *
 * @param [in]     handle            Handle for accessing resources
 * @param [in/out] edgelist          A pointer to the edgelist object
 * @param [in]     list_index        Index of edgelist to update
 * @param [in]     size              Number of entries in list
 * @param [in]     dtype             Datatype for minor vertices
 * @param [out]    minors_array_view Pointer to view of minors array
 * @param [out]    error             Pointer to an error object storing details of any error.  Will
 *                                   be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_minor_vertices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** minors_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add offsets to edgelist object.  A CSR is only supported with
 *            and edgelist of size 1, so the list_index is assumed to be zero.
 *
 * Note that an edgelist should be created either with offsets/indices or with
 * majors/minors, specifying both is an error.
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] edgelist           A pointer to the edgelist object
 * @param [in]     dtype              Datatype for offsets
 * @param [in]     size               Number of entries in list
 * @param [out]    offsets_array_view Pointer to view of majors array
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_offsets(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  cugraph_data_type_id_t dtype,
  size_t size,
  cugraph_type_erased_device_array_view_t** offsets_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add indices to edgelist object.  A CSR is only supported with
 *            an edgelist of size 1, so the list_index is assumed to be zero.
 *
 * Note that an edgelist should be created either with offsets/indices or with
 * majors/minors, specifying both is an error.
 *
 * @param [in]     handle            Handle for accessing resources
 * @param [in/out] edgelist          A pointer to the edgelist object
 * @param [in]     dtype             Datatype for minor vertices
 * @param [in]     size              Number of entries in list
 * @param [out]    minors_array_view Pointer to view of minors array
 * @param [out]    error             Pointer to an error object storing details of any error.  Will
 *                                   be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_indices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  cugraph_data_type_id_t dtype,
  size_t size,
  cugraph_type_erased_device_array_view_t** indices_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add weights to edgelist object.
 *
 * @param [in]     handle             Handle for accessing resources
 * @param [in/out] edgelist           A pointer to the edgelist object
 * @param [in]     list_index         Index of edgelist to update
 * @param [in]     size               Number of entries in list
 * @param [in]     dtype              Datatype for minor vertices
 * @param [out]    weights_array_view Pointer to view of weights array
 * @param [out]    error              Pointer to an error object storing details of any error.  Will
 *                                    be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_weights(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** weights_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add edge ids to edgelist object.
 *
 * @param [in]     handle         Handle for accessing resources
 * @param [in/out] edgelist       A pointer to the edgelist object
 * @param [in]     list_index     Index of edgelist to update
 * @param [in]     size           Number of entries in list
 * @param [in]     dtype          Datatype for minor vertices
 * @param [out]    ids_array_view Pointer to view of ids array
 * @param [out]    error          Pointer to an error object storing details of any error.  Will
 *                                be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_edge_ids(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** ids_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add edge types to edgelist object.
 *
 * @param [in]     handle           Handle for accessing resources
 * @param [in/out] edgelist         A pointer to the edgelist object
 * @param [in]     list_index       Index of edgelist to update
 * @param [in]     size             Number of entries in list
 * @param [in]     dtype            Datatype for minor vertices
 * @param [out]    types_array_view Pointer to view of types array
 * @param [out]    error            Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_edge_types(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** types_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add edge start times to edgelist object.
 *
 * @param [in]     handle           Handle for accessing resources
 * @param [in/out] edgelist         A pointer to the edgelist object
 * @param [in]     list_index       Index of edgelist to update
 * @param [in]     size             Number of entries in list
 * @param [in]     dtype            Datatype for minor vertices
 * @param [out]    times_array_view Pointer to view of times array
 * @param [out]    error            Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_edge_start_times(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** times_array_view,
  cugraph_error_t** error);

/**
 * @brief     Add edge end times to edgelist object.
 *
 * @param [in]     handle           Handle for accessing resources
 * @param [in/out] edgelist         A pointer to the edgelist object
 * @param [in]     list_index       Index of edgelist to update
 * @param [in]     size             Number of entries in list
 * @param [in]     dtype            Datatype for minor vertices
 * @param [out]    times_array_view Pointer to view of times array
 * @param [out]    error            Pointer to an error object storing details of any error.  Will
 *                                  be populated if error code is not CUGRAPH_SUCCESS
 *
 * @return error code
 */
cugraph_error_code_t cugraph_edgelist_add_edge_end_times(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** times_array_view,
  cugraph_error_t** error);

/**
 * @brief     Destroy an edgelist
 *
 * @param [in]  edgelist  A pointer to the edgelist object to destroy
 */
void cugraph_edgelist_free(cugraph_edgelist_t* edgelist);

#ifdef __cplusplus
}
#endif

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

#include "c_api/edgelist.hpp"

#include "c_api/error.hpp"
#include "c_api/resource_handle.hpp"

#include <cugraph_c/edgelist.h>

extern "C" cugraph_error_code_t cugraph_edgelist_create(const cugraph_resource_handle_t* handle,
                                                        size_t num_lists,
                                                        bool_t store_transposed,
                                                        cugraph_edgelist_t** edgelist,
                                                        cugraph_error_t** error)
{
  *edgelist = reinterpret_cast<cugraph_edgelist_t*>(
    new cugraph::c_api::cugraph_edgelist_t{num_lists, store_transposed});
  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_edgelist_set_num_edges(cugraph_edgelist_t* edgelist,
                                                               size_t list_index,
                                                               size_t num_edges,
                                                               cugraph_error_t** error)
{
  auto p_edgelist        = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);
  p_edgelist->num_edges_ = num_edges;
  return CUGRAPH_SUCCESS;
}

namespace {

cugraph_error_code_t add_list(cugraph_resource_handle_t const* handle,
                              std::vector<cugraph_type_erased_device_array_t*>& list,
                              size_t list_index,
                              size_t num_lists,
                              size_t size,
                              cugraph_data_type_id_t dtype,
                              cugraph_type_erased_device_array_view_t** view,
                              cugraph_error_t** error)
{
  if (list.size() == 0) {
    list.resize(num_lists);
    std::fill(list.begin(), list.end(), nullptr);
  }

  CAPI_EXPECTS(list_index < list.size(),
               CUGRAPH_INVALID_INPUT,
               "list_index must be less than num_lists",
               *error);
  CAPI_EXPECTS(list[list_index] == nullptr,
               CUGRAPH_INVALID_INPUT,
               "Tried to create an entry that already exists",
               *error);

  auto error_code =
    cugraph_type_erased_device_array_create(handle, size, dtype, &list[list_index], error);
  if (error_code == CUGRAPH_SUCCESS) {
    *view = cugraph_type_erased_device_array_view(list[list_index]);
  }

  return error_code;
}

}  // namespace

extern "C" cugraph_error_code_t cugraph_edgelist_add_vertices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t num_vertices,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** vertices_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS((p_edgelist->vertex_type_ == NTYPES) || (p_edgelist->vertex_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create vertex array of wrong type",
               *error);

  p_edgelist->vertex_type_ = dtype;

  return add_list(handle,
                  p_edgelist->vertices_,
                  list_index,
                  p_edgelist->num_lists_,
                  num_vertices,
                  dtype,
                  vertices_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_major_vertices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** majors_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS((p_edgelist->vertex_type_ == NTYPES) || (p_edgelist->vertex_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create majors array of wrong type",
               *error);

  CAPI_EXPECTS(!p_edgelist->offsets_ && !p_edgelist->indices_,
               CUGRAPH_INVALID_INPUT,
               "Cannot specify both offsets/indices and majors/minors",
               *error);

  p_edgelist->vertex_type_ = dtype;

  return add_list(handle,
                  p_edgelist->majors_,
                  list_index,
                  p_edgelist->num_lists_,
                  size,
                  dtype,
                  majors_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_minor_vertices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** minors_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add edge times to unintialized edgelist",
               *error);

  CAPI_EXPECTS((p_edgelist->vertex_type_ == NTYPES) || (p_edgelist->vertex_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create minors array of wrong type",
               *error);

  CAPI_EXPECTS(!p_edgelist->offsets_ && !p_edgelist->indices_,
               CUGRAPH_INVALID_INPUT,
               "Cannot specify both offsets/indices and majors/minors",
               *error);

  p_edgelist->vertex_type_ = dtype;

  return add_list(handle,
                  p_edgelist->minors_,
                  list_index,
                  p_edgelist->num_lists_,
                  size,
                  dtype,
                  minors_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_offsets(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  cugraph_data_type_id_t dtype,
  size_t size,
  cugraph_type_erased_device_array_view_t** offsets_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add offsets to unintialized edgelist",
               *error);

  CAPI_EXPECTS(!p_edgelist->offsets_,
               CUGRAPH_INVALID_INPUT,
               "Trying to add offsets after already initialized",
               *error);

  CAPI_EXPECTS(p_edgelist->num_lists_ == 1,
               CUGRAPH_INVALID_INPUT,
               "edgelist with offsets/indices can only have one list",
               *error);

  CAPI_EXPECTS((p_edgelist->majors_.size() == 0) && (p_edgelist->minors_.size() == 0),
               CUGRAPH_INVALID_INPUT,
               "Cannot specify both offsets/indices and majors/minors",
               *error);

  cugraph_type_erased_device_array_t* array;
  auto error_code = cugraph_type_erased_device_array_create(handle, size, dtype, &array, error);

  if (error_code == CUGRAPH_SUCCESS) {
    p_edgelist->offsets_.reset(array);
    *offsets_array_view = cugraph_type_erased_device_array_view(array);
  }

  return error_code;
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_indices(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  cugraph_data_type_id_t dtype,
  size_t size,
  cugraph_type_erased_device_array_view_t** indices_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add indices to unintialized edgelist",
               *error);

  CAPI_EXPECTS(!p_edgelist->indices_,
               CUGRAPH_INVALID_INPUT,
               "Trying to add indices after already initialized",
               *error);

  CAPI_EXPECTS(p_edgelist->num_lists_ == 1,
               CUGRAPH_INVALID_INPUT,
               "edgelist with offsets/indices can only have one list",
               *error);

  CAPI_EXPECTS((p_edgelist->majors_.size() == 0) && (p_edgelist->minors_.size() == 0),
               CUGRAPH_INVALID_INPUT,
               "Cannot specify both offsets/indices and majors/minors",
               *error);

  cugraph_type_erased_device_array_t* array;
  auto error_code = cugraph_type_erased_device_array_create(handle, size, dtype, &array, error);

  if (error_code == CUGRAPH_SUCCESS) {
    p_edgelist->indices_.reset(array);
    *indices_array_view = cugraph_type_erased_device_array_view(array);
  }

  return error_code;
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_weights(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** weights_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add edge times to unintialized edgelist",
               *error);

  CAPI_EXPECTS(
    (p_edgelist->edge_weight_type_ == NTYPES) || (p_edgelist->edge_weight_type_ == dtype),
    CUGRAPH_INVALID_INPUT,
    "Trying to create weights array of wrong type",
    *error);

  p_edgelist->edge_weight_type_ = dtype;

  return add_list(handle,
                  p_edgelist->weights_,
                  list_index,
                  p_edgelist->num_lists_,
                  size,
                  dtype,
                  weights_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_edge_ids(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** ids_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add edge times to unintialized edgelist",
               *error);

  CAPI_EXPECTS((p_edgelist->edge_id_type_ == NTYPES) || (p_edgelist->edge_id_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create edge id array of wrong type",
               *error);

  p_edgelist->edge_id_type_ = dtype;

  return add_list(handle,
                  p_edgelist->edge_ids_,
                  list_index,
                  p_edgelist->num_lists_,
                  size,
                  dtype,
                  ids_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_edge_types(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** types_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add edge times to unintialized edgelist",
               *error);

  CAPI_EXPECTS((p_edgelist->edge_type_type_ == NTYPES) || (p_edgelist->edge_type_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create edge types array of wrong type",
               *error);

  p_edgelist->edge_type_type_ = dtype;

  return add_list(handle,
                  p_edgelist->edge_types_,
                  list_index,
                  p_edgelist->num_lists_,
                  size,
                  dtype,
                  types_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_edge_start_times(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** times_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add edge times to unintialized edgelist",
               *error);

  CAPI_EXPECTS((p_edgelist->edge_time_type_ == NTYPES) || (p_edgelist->edge_time_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create edge start time array of wrong type",
               *error);

  p_edgelist->edge_time_type_ = dtype;

  return add_list(handle,
                  p_edgelist->edge_start_times_,
                  list_index,
                  p_edgelist->num_lists_,
                  size,
                  dtype,
                  times_array_view,
                  error);
}

extern "C" cugraph_error_code_t cugraph_edgelist_add_edge_end_times(
  const cugraph_resource_handle_t* handle,
  cugraph_edgelist_t* edgelist,
  size_t list_index,
  size_t size,
  cugraph_data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** times_array_view,
  cugraph_error_t** error)
{
  auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);

  CAPI_EXPECTS(p_edgelist != nullptr,
               CUGRAPH_INVALID_INPUT,
               "Trying to add edge times to unintialized edgelist",
               *error);

  CAPI_EXPECTS((p_edgelist->edge_time_type_ == NTYPES) || (p_edgelist->edge_time_type_ == dtype),
               CUGRAPH_INVALID_INPUT,
               "Trying to create edge end times array of wrong type",
               *error);

  p_edgelist->edge_time_type_ = dtype;

  return add_list(handle,
                  p_edgelist->edge_end_times_,
                  list_index,
                  p_edgelist->num_lists_,
                  // TODO: In all add_list calls for edge properties... need to keep
                  //   overall size of edges in an std::vector and check that sizes match
                  size,
                  dtype,
                  times_array_view,
                  error);
}

extern "C" void cugraph_edgelist_free(cugraph_edgelist_t* edgelist)
{
  if (edgelist != nullptr) {
    auto p_edgelist = reinterpret_cast<cugraph::c_api::cugraph_edgelist_t*>(edgelist);
    delete p_edgelist;
    edgelist = nullptr;
  }
}

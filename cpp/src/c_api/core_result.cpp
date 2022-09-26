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

#include <cugraph_c/algorithms.h>

#include <c_api/core_result.hpp>

extern "C" cugraph_error_code_t cugraph_core_result_create(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_view_t* vertices,
  cugraph_type_erased_device_array_view_t* core_numbers,
  cugraph_core_result_t** core_result,
  cugraph_error_t** error)
{
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  cugraph::c_api::cugraph_type_erased_device_array_t* vertices_copy;
  cugraph::c_api::cugraph_type_erased_device_array_t* core_numbers_copy;

  error_code = cugraph_type_erased_device_array_create_from_view(
    handle,
    vertices,
    reinterpret_cast<cugraph_type_erased_device_array_t**>(&vertices_copy),
    error);
  if (error_code == CUGRAPH_SUCCESS) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle,
      core_numbers,
      reinterpret_cast<cugraph_type_erased_device_array_t**>(&core_numbers_copy),
      error);

    if (error_code == CUGRAPH_SUCCESS) {
      auto internal_pointer =
        new cugraph::c_api::cugraph_core_result_t{vertices_copy, core_numbers_copy};
      *core_result = reinterpret_cast<cugraph_core_result_t*>(internal_pointer);
    }
  }
  return error_code;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_core_result_get_vertices(
  cugraph_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_core_result_get_core_numbers(
  cugraph_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->core_numbers_->view());
}

extern "C" void cugraph_core_result_free(cugraph_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_core_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->core_numbers_;
  delete internal_pointer;
}

cugraph_type_erased_device_array_view_t* cugraph_k_core_result_get_src_vertices(
  cugraph_k_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_k_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->src_vertices_->view());
}

cugraph_type_erased_device_array_view_t* cugraph_k_core_result_get_dst_vertices(
  cugraph_k_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_k_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->dst_vertices_->view());
}

cugraph_type_erased_device_array_view_t* cugraph_k_core_result_get_weights(
  cugraph_k_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_k_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->weights_->view());
}

void cugraph_k_core_result_free(cugraph_k_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_k_core_result_t*>(result);
  delete internal_pointer->src_vertices_;
  delete internal_pointer->dst_vertices_;
  delete internal_pointer->weights_;
  delete internal_pointer;
}

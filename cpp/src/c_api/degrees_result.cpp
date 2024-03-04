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

#include "c_api/degrees_result.hpp"

#include <cugraph_c/graph_functions.h>

extern "C" cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_vertices(
  cugraph_degrees_result_t* degrees_result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_degrees_result_t*>(degrees_result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_in_degrees(
  cugraph_degrees_result_t* degrees_result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_degrees_result_t*>(degrees_result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->in_degrees_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_out_degrees(
  cugraph_degrees_result_t* degrees_result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_degrees_result_t*>(degrees_result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->out_degrees_->view());
}

extern "C" void cugraph_degrees_result_free(cugraph_degrees_result_t* degrees_result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_degrees_result_t*>(degrees_result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->in_degrees_;
  delete internal_pointer->out_degrees_;
  delete internal_pointer;
}

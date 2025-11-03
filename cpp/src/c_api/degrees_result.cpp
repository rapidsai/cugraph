/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  return internal_pointer->in_degrees_ == nullptr
           ? nullptr
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->in_degrees_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_degrees_result_get_out_degrees(
  cugraph_degrees_result_t* degrees_result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_degrees_result_t*>(degrees_result);
  return internal_pointer->out_degrees_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->out_degrees_->view())
         : internal_pointer->is_symmetric
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->in_degrees_->view())
           : nullptr;
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

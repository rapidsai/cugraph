/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/labeling_result.hpp"

#include <cugraph_c/labeling_algorithms.h>

extern "C" cugraph_type_erased_device_array_view_t* cugraph_labeling_result_get_vertices(
  cugraph_labeling_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_labeling_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_labeling_result_get_labels(
  cugraph_labeling_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_labeling_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->labels_->view());
}

extern "C" void cugraph_labeling_result_free(cugraph_labeling_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_labeling_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->labels_;
  delete internal_pointer;
}

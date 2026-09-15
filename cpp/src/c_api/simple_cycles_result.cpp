/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/simple_cycles_result.hpp"

#include <cugraph_c/components_algorithms.h>

extern "C" cugraph_type_erased_device_array_view_t* cugraph_simple_cycles_result_get_cycle_vertices(
  cugraph_simple_cycles_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_simple_cycles_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->cycle_vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_simple_cycles_result_get_cycle_offsets(
  cugraph_simple_cycles_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_simple_cycles_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->cycle_offsets_->view());
}

extern "C" size_t cugraph_simple_cycles_result_get_num_cycles(
  cugraph_simple_cycles_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_simple_cycles_result_t*>(result);
  return internal_pointer->cycle_offsets_->size_ - 1;
}

extern "C" void cugraph_simple_cycles_result_free(cugraph_simple_cycles_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_simple_cycles_result_t*>(result);
  delete internal_pointer->cycle_vertices_;
  delete internal_pointer->cycle_offsets_;
  delete internal_pointer;
}

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

#include <c_api/centrality_result.hpp>

extern "C" cugraph_type_erased_device_array_view_t* cugraph_centrality_result_get_vertices(
  cugraph_centrality_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_centrality_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_centrality_result_get_values(
  cugraph_centrality_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_centrality_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->values_->view());
}

extern "C" void cugraph_centrality_result_free(cugraph_centrality_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_centrality_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->values_;
  delete internal_pointer;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_edge_centrality_result_get_src_vertices(
  cugraph_edge_centrality_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_centrality_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->src_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_edge_centrality_result_get_dst_vertices(
  cugraph_edge_centrality_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_centrality_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->dst_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_edge_centrality_result_get_values(
  cugraph_edge_centrality_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_centrality_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->values_->view());
}

extern "C" void cugraph_edge_centrality_result_free(cugraph_edge_centrality_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_centrality_result_t*>(result);
  delete internal_pointer->src_ids_;
  delete internal_pointer->dst_ids_;
  delete internal_pointer->values_;
  delete internal_pointer;
}

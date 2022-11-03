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

#include <c_api/induced_subgraph_result.hpp>

extern "C" cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_sources(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->src_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_destinations(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->dst_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_edge_weights(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->wgt_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_subgraph_offsets(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->subgraph_offsets_->view());
}

extern "C" void cugraph_induced_subgraph_result_free(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  delete internal_pointer->src_;
  delete internal_pointer->dst_;
  delete internal_pointer->wgt_;
  delete internal_pointer->subgraph_offsets_;
  delete internal_pointer;
}

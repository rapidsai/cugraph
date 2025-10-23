/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/induced_subgraph_result.hpp"

#include <cugraph_c/algorithms.h>

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
  return (internal_pointer->wgt_ == nullptr)
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->wgt_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_edge_ids(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  return (internal_pointer->edge_ids_ == nullptr)
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_induced_subgraph_get_edge_type_ids(
  cugraph_induced_subgraph_result_t* induced_subgraph)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t*>(induced_subgraph);
  return (internal_pointer->edge_type_ids_ == nullptr)
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_type_ids_->view());
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
  delete internal_pointer->edge_ids_;
  delete internal_pointer->edge_type_ids_;
  delete internal_pointer->subgraph_offsets_;
  delete internal_pointer;
}

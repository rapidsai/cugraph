/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/centrality_result.hpp"

#include <cugraph_c/algorithms.h>

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

size_t cugraph_centrality_result_get_num_iterations(cugraph_centrality_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_centrality_result_t*>(result);
  return internal_pointer->num_iterations_;
}

bool_t cugraph_centrality_result_converged(cugraph_centrality_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_centrality_result_t*>(result);
  return internal_pointer->converged_ ? bool_t::TRUE : bool_t::FALSE;
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

extern "C" cugraph_type_erased_device_array_view_t* cugraph_edge_centrality_result_get_edge_ids(
  cugraph_edge_centrality_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_centrality_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->edge_ids_->view());
}

extern "C" void cugraph_edge_centrality_result_free(cugraph_edge_centrality_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_centrality_result_t*>(result);
  delete internal_pointer->src_ids_;
  delete internal_pointer->dst_ids_;
  delete internal_pointer->values_;
  delete internal_pointer->edge_ids_;
  delete internal_pointer;
}

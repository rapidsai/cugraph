/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/hierarchical_clustering_result.hpp"

#include <cugraph_c/community_algorithms.h>

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_hierarchical_clustering_result_get_vertices(
  cugraph_hierarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_hierarchical_clustering_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_hierarchical_clustering_result_get_clusters(
  cugraph_hierarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_hierarchical_clustering_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->clusters_->view());
}

extern "C" double cugraph_hierarchical_clustering_result_get_modularity(
  cugraph_hierarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_hierarchical_clustering_result_t*>(result);
  return internal_pointer->modularity;
}

extern "C" void cugraph_hierarchical_clustering_result_free(
  cugraph_hierarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_hierarchical_clustering_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->clusters_;
  delete internal_pointer;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "community/detail/common_methods.hpp"

#include <cugraph/dendrogram.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/vertex_property.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> partition_at_level(raft::handle_t const& handle,
                                                 dendrogram_t<vertex_t> const& dendrogram,
                                                 size_t num_levels)
{
  rmm::device_uvector<vertex_t> partition(dendrogram.level_size(0), handle.get_stream());

  cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                    partition.begin(),
                    partition.end(),
                    dendrogram.level_range_first(0));

  std::for_each(thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(num_levels),
                [&handle, &dendrogram, &partition](size_t l) {
                  rmm::device_uvector<vertex_t> local_vertex_ids_v(dendrogram.level_size(l),
                                                                   handle.get_stream());

                  cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                                    local_vertex_ids_v.begin(),
                                    local_vertex_ids_v.end(),
                                    dendrogram.level_range_first(l));

                  cugraph::relabel<vertex_t, multi_gpu>(
                    handle,
                    std::tuple<vertex_t const*, vertex_t const*>(local_vertex_ids_v.data(),
                                                                 dendrogram.level_data(l).data()),
                    dendrogram.level_size(l),
                    partition.data(),
                    static_cast<vertex_t>(dendrogram.level_size(0)),
                    false);
                });

  return partition;
}

template <typename vertex_t, bool multi_gpu>
void relabel_cluster_ids(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& unique_cluster_ids,
                         vertex_t* clustering,
                         size_t num_nodes)
{
  vertex_t local_cluster_id_first{0};

  remove_duplicates<vertex_t, multi_gpu>(handle, unique_cluster_ids);

  if constexpr (multi_gpu) {
    auto cluster_ids_size_per_rank = cugraph::host_scalar_allgather(
      handle.get_comms(), unique_cluster_ids.size(), handle.get_stream());

    std::vector<vertex_t> cluster_ids_starts(cluster_ids_size_per_rank.size());
    std::exclusive_scan(cluster_ids_size_per_rank.begin(),
                        cluster_ids_size_per_rank.end(),
                        cluster_ids_starts.begin(),
                        size_t{0});
    local_cluster_id_first = cluster_ids_starts[handle.get_comms().get_rank()];
  }

  rmm::device_uvector<vertex_t> numbering_indices(unique_cluster_ids.size(), handle.get_stream());
  cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                    numbering_indices.data(),
                    numbering_indices.data() + numbering_indices.size(),
                    local_cluster_id_first);

  relabel<vertex_t, multi_gpu>(
    handle,
    std::make_tuple(static_cast<vertex_t const*>(unique_cluster_ids.begin()),
                    static_cast<vertex_t const*>(numbering_indices.begin())),
    unique_cluster_ids.size(),
    clustering,
    num_nodes,
    false);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
vertex_property_t<vertex_t, vertex_t> flatten_dendrogram(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  dendrogram_t<vertex_t> const& dendrogram,
  bool compact_cluster_ids)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  auto cluster_ids =
    detail::partition_at_level<vertex_t, multi_gpu>(handle, dendrogram, dendrogram.num_levels());
  if (compact_cluster_ids) {
    rmm::device_uvector<vertex_t> unique_cluster_ids(
      static_cast<size_t>(graph_view.local_vertex_partition_range_size()), handle.get_stream());
    copy_v(handle,
           graph_view,
           vertex_property_view_t<vertex_t, vertex_t const*, vertex_t>{
             cluster_ids.data(), static_cast<vertex_t>(cluster_ids.size())},
           vertex_property_view_t<vertex_t, vertex_t*, vertex_t>{
             unique_cluster_ids.data(), graph_view.local_vertex_partition_range_size()});
    detail::relabel_cluster_ids<vertex_t, multi_gpu>(
      handle, unique_cluster_ids, cluster_ids.data(), cluster_ids.size());
  }
  auto const num_verts = static_cast<vertex_t>(cluster_ids.size());

  return vertex_property_t<vertex_t, vertex_t>(std::move(cluster_ids), num_verts);
}

template <typename graph_view_t>
void flatten_dendrogram(raft::handle_t const& handle,
                        graph_view_t const& graph_view,
                        dendrogram_t<typename graph_view_t::vertex_type> const& dendrogram,
                        typename graph_view_t::vertex_type* clustering)
{
  static_assert(!graph_view_t::is_storage_transposed,
                "flatten_dendrogram expects the graph view to be untransposed.");

  auto clustering_prop = cugraph::flatten_dendrogram(handle, graph_view, dendrogram, false);
  using vertex_t       = typename graph_view_t::vertex_type;
  copy_v(handle,
         graph_view,
         clustering_prop.view(),
         vertex_property_view_t<vertex_t, vertex_t*, vertex_t>{
           clustering, graph_view.local_vertex_partition_range_size()});
}

}  // namespace cugraph

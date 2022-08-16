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
#pragma once
#include <community/flatten_dendrogram.cuh>
#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace detail {

// FIXME: Can we have a common check_clustering to be used by both
// Louvain and Leiden, and possibly other clustering methods?
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed=false>
void check_clustering(graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
            vertex_t* clustering)
{
if (graph_view.local_vertex_partition_range_size() > 0)
CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed=false>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t>
leiden(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
       size_t max_level,
       weight_t resolution)
{
  // TODO: everything
  CUGRAPH_FAIL("unimplemented");
  return std::make_pair(std::make_unique<Dendrogram<vertex_t>>(), weight_t{0.0});
}

// FIXME: Can we have a common flatten_dendrogram to be used by both
// Louvain and Leiden, and possibly other clustering methods?
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          bool store_transposed=false>
void flatten_dendrogram(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  Dendrogram<vertex_t> const& dendrogram,
  vertex_t* clustering)
{
  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices(), handle.get_stream());

  thrust::sequence(handle.get_thrust_policy(),
                   vertex_ids_v.begin(),
                   vertex_ids_v.end(),
                   graph_view.local_vertex_partition_range_first());

  partition_at_level<vertex_t, multi_gpu>(
    handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
}

}  // namespace detail

template <typename GraphViewType>
std::pair<std::unique_ptr<Dendrogram<typename GraphViewType::vertex_type>>,
          typename GraphViewType::weight_type>
leiden(raft::handle_t const& handle,
       GraphViewType const& graph_view,
       size_t max_level,
       typename GraphViewType::weight_type resolution)
{
  return detail::leiden(handle, graph_view, max_level, resolution);
}

template <typename GraphViewType>
void flatten_dendrogram(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        Dendrogram<typename GraphViewType::vertex_type> const& dendrogram,
                        typename GraphViewType::vertex_type* clustering)
{
  detail::flatten_dendrogram(handle, graph_view, dendrogram, clustering);
}

}  // namespace cugraph

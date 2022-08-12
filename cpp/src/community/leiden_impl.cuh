/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

template <typename GraphViewType>
void check_clustering(GraphViewType const& graph_view, typename GraphViewType::vertex_t* clustering)
{
  if (graph_view.local_vertex_partition_range_size() > 0)
    CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");
}

template <typename GraphViewType>
std::pair<std::unique_ptr<Dendrogram<typename GraphViewType::vertex_type>>,
          typename GraphViewType::weight_type>
leiden(raft::handle_t const& handle,
       GraphViewType const& graph_view,
       size_t max_level,
       typename GraphViewType::weight_type resolution)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

  // TODO: everything
  CUGRAPH_FAIL("unimplemented");
  return std::make_pair(std::make_unique<Dendrogram<vertex_t>>(), (weight_t)0.0);
}

template <typename GraphViewType>
std::pair<size_t, typename GraphViewType::weight_type> leiden(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  typename GraphViewType::vertex_type* clustering,
  size_t max_level,
  typename GraphViewType::weight_type resolution);

template <typename GraphViewType>
void flatten_dendrogram(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        Dendrogram<typename GraphViewType::vertex_t> const& dendrogram,
                        typename GraphViewType::vertex_t* clustering)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;
  bool multi_gpu = GraphViewType::multi_gpu;

  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(),
                   vertex_ids_v.begin(),
                   vertex_ids_v.end(),
                   graph_view.local_vertex_partition_range_first());
  partition_at_level<vertex_t, multi_gpu>(
    handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
}

}  // namespace detail

// Keep the API similar to Louvain

template <typename GraphViewType>
void flatten_dendrogram(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        Dendrogram<typename GraphViewType::vertex_type> const& dendrogram,
                        typename GraphViewType::vertex_type* clustering)
{
  detail::flatten_dendrogram(handle, graph_view, dendrogram, clustering);
}

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
std::pair<size_t, typename GraphViewType::weight_type> leiden(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  typename GraphViewType::vertex_type* clustering,
  size_t max_level,
  typename GraphViewType::weight_type resolution)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

  CUGRAPH_EXPECTS(graph_view.is_weighted(), "Graph must be weighted");

  detail::check_clustering(graph_view, clustering);

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram;
  weight_t modularity;

  std::tie(dendrogram, modularity) = leiden(handle, graph_view, max_level, resolution);

  flatten_dendrogram(handle, graph_view, *dendrogram, clustering);

  return std::make_pair(dendrogram->num_levels(), modularity);
}

}  // namespace cugraph

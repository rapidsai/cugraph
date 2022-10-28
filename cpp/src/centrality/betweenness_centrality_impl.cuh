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

#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void brandes_bfs(const raft::handle_t& handle,
                 graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                 vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier)
{
  //
  // Do BFS with a multi-output.  If we're on hop k and multiple vertices arrive at vertex v,
  // add all predecessors to the predecessor list, don't just arbitrarily pick one.
  //
  // Predecessors could be a CSR if that's helpful for doing the backwards tracing
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_vertex_results(rmm::device_uvector<weight_t>& centralities,
                               result,
                               bool with_endpoints)
{
  //
  // Traverse back pointers to update centralities
  //
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
rmm::device_uvector<weight_t> betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {}

  rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());

  size_t num_sources = thrust::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_starts{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_local_sources, handle.get_stream());
    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_starts.resize(source_counts.size() + 1);
    source_starts[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_starts.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    //
    //  BFS
    constexpr size_t bucket_idx_cur  = 0;
    constexpr size_t bucket_idx_next = 1;
    constexpr size_t num_buckets     = 2;

    vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(
      handle, num_buckets);

    if ((source_idx >= source_starts[my_rank]) && (source_idx < source_starts[my_rank + 1]))
      vertex_frontier.bucket(bucket_idx_cur)
        .insert(vertices_begin + (source_idx - source_starts[my_rank]),
                vertices_begin + (source_idx - source_starts[my_rank]) + 1);

    //
    //  Now we need to do modified BFS
    //
    auto result = brandes_bfs(handle, graph_view, vertex_frontier);
    accumulate_vertex_results(centralities, result, count_endpoints);
  }

  return centralities;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
rmm::device_uvector<weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const do_expensive_check)
{
  CUGRAPH_FAIL("Not Implemented");
#if 0
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {}

  // FIXME:  Not sure how to compute the number of results here?
  //         Or how to return these.
  //  For now, let's ignore edge betweenness and focus on vertex betweenness.
  //  It might be that we wait until we have edge ids and we can use the edge id
  //  as the basis for an index here.
  rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());

  size_t num_sources = thrust::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_starts{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_local_sources, handle.get_stream());
    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_starts.resize(source_counts.size() + 1);
    source_starts[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_starts.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  for (size_t source_idx = 0 ; source_idx < num_sources ; ++source_idx) {
    //
    //  BFS
    constexpr size_t bucket_idx_cur  = 0;
    constexpr size_t bucket_idx_next = 1;
    constexpr size_t num_buckets     = 2;

    vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                         num_buckets);

    if ((source_idx >= source_starts[my_rank]) && (source_idx < source_starts[my_rank+1]))
      vertex_frontier.bucket(bucket_idx_cur).insert(vertices_begin + (source_idx - source_starts[my_rank]),
                                                    vertices_begin + (source_idx - source_starts[my_rank]) + 1);

    //
    //  Now we need to do modified BFS
    //
    auto result = brandes_bfs(handle,graph_view,vertex_frontier);
    accumulate_edge_centrality(centralities, result);
  }

  return centralities;
#endif
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  std::optional<std::variant<vertex_t, raft::device_span<vertex_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  if (vertices) {
    if (std::hold_alternative<vertex_t>(vertices)) {
      rmm::device_uvector<vertex_t> select_vertices(std::get<vertex_t>(vertices),
                                                    handle.get_stream());
      // Populate with random vertices
      return detail::betweenness_centrality(handle,
                                            graph_view,
                                            select_vertices.begin(),
                                            select_vertices.end(),
                                            normalized,
                                            include_endpoints,
                                            do_expensive_check);
    } else {
      auto provided_vertices = std::get<raft::device_span<vertex_t const>>(vertices),
           return detail::betweenness_centrality(handle,
                                                 graph_view,
                                                 provided_vertices.begin(),
                                                 provided_vertices.end(),
                                                 normalized,
                                                 include_endpoints,
                                                 do_expensive_check);
    }
  } else {
    return detail::betweenness_centrality(
      handle,
      graph_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      include_endpoints,
      do_expensive_check);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  std::optional<std::variant<vertex_t, raft::device_span<vertex_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check)
{
  if (vertices) {
    if (std::hold_alternative<vertex_t>(vertices)) {
      rmm::device_uvector<vertex_t> select_vertices(std::get<vertex_t>(vertices),
                                                    handle.get_stream());
      // Populate with random vertices
      return detail::betweenness_centrality(handle,
                                            graph_view,
                                            select_vertices.begin(),
                                            select_vertices.end(),
                                            normalized,
                                            include_endpoints,
                                            do_expensive_check);
    } else {
      auto provided_vertices = std::get<raft::device_span<vertex_t const>>(vertices),
           return detail::betweenness_centrality(handle,
                                                 graph_view,
                                                 provided_vertices.begin(),
                                                 provided_vertices.end(),
                                                 normalized,
                                                 include_endpoints,
                                                 do_expensive_check);
    }
  } else {
    return detail::betweenness_centrality(
      handle,
      graph_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      include_endpoints,
      do_expensive_check);
  }
}

}  // namespace cugraph

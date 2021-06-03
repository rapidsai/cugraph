/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <community/flatten_dendrogram.cuh>
#include <community/louvain.cuh>
#include <cugraph/experimental/graph.hpp>
#include <experimental/louvain.cuh>

#include <rmm/device_uvector.hpp>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> louvain(
  raft::handle_t const &handle,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph_view,
  size_t max_level,
  weight_t resolution)
{
  CUGRAPH_EXPECTS(graph_view.edge_data != nullptr,
                  "Invalid input argument: louvain expects a weighted graph");

  Louvain<GraphCSRView<vertex_t, edge_t, weight_t>> runner(handle, graph_view);
  weight_t wt = runner(max_level, resolution);

  return std::make_pair(runner.move_dendrogram(), wt);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> louvain(
  raft::handle_t const &handle,
  experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
  size_t max_level,
  weight_t resolution)
{
  experimental::Louvain<experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu>>
    runner(handle, graph_view);

  weight_t wt = runner(max_level, resolution);

  return std::make_pair(runner.move_dendrogram(), wt);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void flatten_dendrogram(raft::handle_t const &handle,
                        GraphCSRView<vertex_t, edge_t, weight_t> const &graph_view,
                        Dendrogram<vertex_t> const &dendrogram,
                        vertex_t *clustering)
{
  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices, handle.get_stream());

  thrust::sequence(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   vertex_ids_v.begin(),
                   vertex_ids_v.end(),
                   vertex_t{0});

  partition_at_level<vertex_t, false>(
    handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void flatten_dendrogram(
  raft::handle_t const &handle,
  experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
  Dendrogram<vertex_t> const &dendrogram,
  vertex_t *clustering)
{
  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.get_number_of_vertices(),
                                             handle.get_stream());

  thrust::sequence(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   vertex_ids_v.begin(),
                   vertex_ids_v.end(),
                   graph_view.get_local_vertex_first());

  partition_at_level<vertex_t, multi_gpu>(
    handle, dendrogram, vertex_ids_v.data(), clustering, dendrogram.num_levels());
}

}  // namespace detail

template <typename graph_view_t>
std::pair<std::unique_ptr<Dendrogram<typename graph_view_t::vertex_type>>,
          typename graph_view_t::weight_type>
louvain(raft::handle_t const &handle,
        graph_view_t const &graph_view,
        size_t max_level,
        typename graph_view_t::weight_type resolution)
{
  return detail::louvain(handle, graph_view, max_level, resolution);
}

template <typename graph_view_t>
void flatten_dendrogram(raft::handle_t const &handle,
                        graph_view_t const &graph_view,
                        Dendrogram<typename graph_view_t::vertex_type> const &dendrogram,
                        typename graph_view_t::vertex_type *clustering)
{
  detail::flatten_dendrogram(handle, graph_view, dendrogram, clustering);
}

template <typename graph_view_t>
std::pair<size_t, typename graph_view_t::weight_type> louvain(
  raft::handle_t const &handle,
  graph_view_t const &graph_view,
  typename graph_view_t::vertex_type *clustering,
  size_t max_level,
  typename graph_view_t::weight_type resolution)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using weight_t = typename graph_view_t::weight_type;

  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram;
  weight_t modularity;

  std::tie(dendrogram, modularity) = louvain(handle, graph_view, max_level, resolution);

  flatten_dendrogram(handle, graph_view, *dendrogram, clustering);

  return std::make_pair(dendrogram->num_levels(), modularity);
}

// Explicit template instantations
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, float, false, false> const &,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, float, false, false> const &,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, float, false, false> const &,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, double, false, false> const &,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, double, false, false> const &,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, double, false, false> const &,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, float, false, true> const &,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, float, false, true> const &,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, float, false, true> const &,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, double, false, true> const &,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, double, false, true> const &,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, double, false, true> const &,
  size_t,
  double);

template std::pair<size_t, float> louvain(
  raft::handle_t const &, GraphCSRView<int32_t, int32_t, float> const &, int32_t *, size_t, float);
template std::pair<size_t, double> louvain(raft::handle_t const &,
                                           GraphCSRView<int32_t, int32_t, double> const &,
                                           int32_t *,
                                           size_t,
                                           double);
template std::pair<size_t, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, float, false, false> const &,
  int32_t *,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, double, false, false> const &,
  int32_t *,
  size_t,
  double);
template std::pair<size_t, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, float, false, false> const &,
  int32_t *,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, double, false, false> const &,
  int32_t *,
  size_t,
  double);
template std::pair<size_t, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, float, false, false> const &,
  int64_t *,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, double, false, false> const &,
  int64_t *,
  size_t,
  double);

// instantations with multi_gpu = true
template std::pair<size_t, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, float, false, true> const &,
  int32_t *,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int32_t, double, false, true> const &,
  int32_t *,
  size_t,
  double);

template std::pair<size_t, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, float, false, true> const &,
  int32_t *,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int32_t, int64_t, double, false, true> const &,
  int32_t *,
  size_t,
  double);
template std::pair<size_t, float> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, float, false, true> const &,
  int64_t *,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const &,
  experimental::graph_view_t<int64_t, int64_t, double, false, true> const &,
  int64_t *,
  size_t,
  double);

}  // namespace cugraph

#include <cugraph/eidir_graph.hpp>

/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <community/louvain.cuh>

// "FIXME": remove the guards after support for Pascal will be dropped;
//
// Disable louvain(experimenta::graph_view_t,...)
// versions for GPU architectures < 700
//(this is because cuco/static_map.cuh would not
// compile on those)
//
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#include <experimental/graph.hpp>
#else
#include <experimental/louvain.cuh>
#endif

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<size_t, weight_t> louvain(raft::handle_t const &handle,
                                    GraphCSRView<vertex_t, edge_t, weight_t> const &graph_view,
                                    vertex_t *clustering,
                                    size_t max_level,
                                    weight_t resolution)
{
  CUGRAPH_EXPECTS(graph_view.edge_data != nullptr,
                  "Invalid input argument: louvain expects a weighted graph");
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");

  Louvain<GraphCSRView<vertex_t, edge_t, weight_t>> runner(handle, graph_view);
  return runner(clustering, max_level, resolution);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<size_t, weight_t> louvain(
  raft::handle_t const &handle,
  experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
  vertex_t *clustering,
  size_t max_level,
  weight_t resolution)
{
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
  CUGRAPH_FAIL("Louvain not supported on Pascal and older architectures");
#else
  experimental::Louvain<experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu>>
    runner(handle, graph_view);
  return runner(clustering, max_level, resolution);
#endif
}

}  // namespace detail

template <typename graph_t>
std::pair<size_t, typename graph_t::weight_type> louvain(raft::handle_t const &handle,
                                                         graph_t const &graph,
                                                         typename graph_t::vertex_type *clustering,
                                                         size_t max_level,
                                                         typename graph_t::weight_type resolution)
{
  CUGRAPH_EXPECTS(clustering != nullptr, "Invalid input argument: clustering is null");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
  CUGRAPH_FAIL("Louvain not supported on Pascal and older architectures");
#else
  return detail::louvain(handle, graph, clustering, max_level, resolution);
#endif
}

// Explicit template instantations
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

#include <eidir_graph.hpp>

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

#pragma once

#include <community/flatten_dendrogram.cuh>
#include <community/louvain.cuh>
#include <experimental/graph.hpp>
#include <experimental/louvain.cuh>

#include <rmm/device_uvector.hpp>

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
  CUGRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a device pointer to "
                  "memory for storing the result");

  Louvain<GraphCSRView<vertex_t, edge_t, weight_t>> runner(handle, graph_view);
  weight_t wt = runner(max_level, resolution);

  rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.number_of_vertices, handle.get_stream());

  thrust::sequence(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   vertex_ids_v.begin(),
                   vertex_ids_v.end(),
                   vertex_t{0});

  partition_at_level<vertex_t, false>(handle,
                                      runner.get_dendrogram(),
                                      vertex_ids_v.data(),
                                      clustering,
                                      runner.get_dendrogram().num_levels());

  // FIXME: Consider returning the Dendrogram at some point
  return std::make_pair(runner.get_dendrogram().num_levels(), wt);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<size_t, weight_t> louvain(
  raft::handle_t const &handle,
  experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
  vertex_t *clustering,
  size_t max_level,
  weight_t resolution)
{
  CUGRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a device pointer to "
                  "memory for storing the result");

  // "FIXME": remove this check and the guards below
  //
  // Disable louvain(experimental::graph_view_t,...)
  // versions for GPU architectures < 700
  // (cuco/static_map.cuh depends on features not supported on or before Pascal)
  //
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

  if (device_prop.major < 7) {
    CUGRAPH_FAIL("Louvain not supported on Pascal and older architectures");
  } else {
    experimental::Louvain<experimental::graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu>>
      runner(handle, graph_view);

    weight_t wt = runner(max_level, resolution);

    rmm::device_uvector<vertex_t> vertex_ids_v(graph_view.get_number_of_vertices(),
                                               handle.get_stream());

    thrust::sequence(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     vertex_ids_v.begin(),
                     vertex_ids_v.end(),
                     graph_view.get_local_vertex_first());

    partition_at_level<vertex_t, multi_gpu>(handle,
                                            runner.get_dendrogram(),
                                            vertex_ids_v.data(),
                                            clustering,
                                            runner.get_dendrogram().num_levels());

    // FIXME: Consider returning the Dendrogram at some point
    return std::make_pair(runner.get_dendrogram().num_levels(), wt);
  }
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

  return detail::louvain(handle, graph, clustering, max_level, resolution);
}

}  // namespace cugraph

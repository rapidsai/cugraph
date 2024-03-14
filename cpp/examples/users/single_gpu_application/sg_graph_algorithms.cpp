/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <iostream>
#include <string>

std::unique_ptr<raft::handle_t> initialize_sg_handle()
{
  RAFT_CUDA_TRY(cudaSetDevice(0));
  std::shared_ptr<rmm::mr::device_memory_resource> resource =
    std::make_shared<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_current_device_resource(resource.get());

  std::unique_ptr<raft::handle_t> handle =
    std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread, resource);
  return std::move(handle);
}

/**
 * @brief Create a graph from edge sources, destinitions, and optional weights
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>

std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<cugraph::edge_property_t<
             cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
             weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
create_graph(raft::handle_t const& handle,
             std::vector<vertex_t>&& edge_srcs,
             std::vector<vertex_t>&& edge_dsts,
             std::optional<std::vector<weight_t>>&& edge_wgts,
             bool renumber,
             bool is_symmetric)
{
  size_t num_edges = edge_srcs.size();
  assert(edge_dsts.size() == num_edges);
  if (edge_wgts.has_value()) { assert((*edge_wgts).size() == num_edges); }

  std::cout << "#E = " << num_edges << std::endl;

  rmm::device_uvector<vertex_t> d_edge_srcs(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_edge_dsts(num_edges, handle.get_stream());

  auto d_edge_wgts = edge_wgts.has_value() ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               num_edges, handle.get_stream())
                                           : std::nullopt;

  raft::update_device(d_edge_srcs.data(), edge_srcs.data(), num_edges, handle.get_stream());
  raft::update_device(d_edge_dsts.data(), edge_dsts.data(), num_edges, handle.get_stream());
  if (d_edge_wgts.has_value()) {
    raft::update_device((*d_edge_wgts).data(), (*edge_wgts).data(), num_edges, handle.get_stream());
  }

  //
  // Create graph
  //

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);

  std::optional<cugraph::edge_property_t<decltype(graph.view()), weight_t>> edge_weights{
    std::nullopt};

  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, std::ignore, std::ignore, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_t,
                                        int32_t,
                                        store_transposed,
                                        multi_gpu>(handle,
                                                   std::nullopt,
                                                   std::move(d_edge_srcs),
                                                   std::move(d_edge_dsts),
                                                   std::move(d_edge_wgts),
                                                   std::nullopt,
                                                   std::nullopt,
                                                   cugraph::graph_properties_t{is_symmetric, false},
                                                   renumber,
                                                   true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

/**
 * @brief Run BFS and Louvain on input graph.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
void run_graph_algorithms(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  //
  // BFS
  //

  rmm::device_uvector<vertex_t> d_distances(graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());

  rmm::device_uvector<vertex_t> d_sources(1, handle.get_stream());
  std::vector<vertex_t> h_sources = {0};
  raft::update_device(d_sources.data(), h_sources.data(), h_sources.size(), handle.get_stream());

  cugraph::bfs(handle,
               graph_view,
               d_distances.data(),
               d_predecessors.data(),
               d_sources.data(),
               d_sources.size(),
               false,
               std::numeric_limits<vertex_t>::max());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  auto distances_title = std::string("distances");
  raft::print_device_vector(
    distances_title.c_str(), d_distances.begin(), d_distances.size(), std::cout);

  auto predecessors_title = std::string("predecessors");
  raft::print_device_vector(
    predecessors_title.c_str(), d_predecessors.begin(), d_predecessors.size(), std::cout);

  //
  // Louvain
  //

  rmm::device_uvector<vertex_t> d_cluster_assignments(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());

  weight_t threshold  = 1e-7;
  weight_t resolution = 1.0;
  size_t max_level    = 10;

  weight_t modularity{-1.0};
  std::tie(std::ignore, modularity) =
    cugraph::louvain(handle,
                     std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
                     graph_view,
                     edge_weight_view,
                     d_cluster_assignments.data(),
                     max_level,
                     threshold,
                     resolution);

  std::cout << "modularity : " << modularity << std::endl;

  auto cluster_assignments_title = std::string("cluster_assignments");
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  raft::print_device_vector(cluster_assignments_title.c_str(),
                            d_cluster_assignments.begin(),
                            d_cluster_assignments.size(),
                            std::cout);
}

int main(int argc, char** argv)
{
  std::unique_ptr<raft::handle_t> handle = initialize_sg_handle();

  //
  // Create graph from edge source, destinition and weight list
  //

  using vertex_t    = int32_t;
  using edge_t      = int32_t;
  using weight_t    = float;
  bool is_symmetric = true;

  std::vector<vertex_t> edge_srcs = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<vertex_t> edge_dsts = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<weight_t> edge_wgts = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  constexpr bool multi_gpu        = false;
  constexpr bool store_transposed = false;
  bool renumber                   = true;

  auto [graph, edge_weights, renumber_map] =
    create_graph<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      *handle,
      std::move(edge_srcs),
      std::move(edge_dsts),
      std::move(std::make_optional(edge_wgts)),
      renumber,
      is_symmetric);

  // Non-owning view of the graph object
  auto graph_view = graph.view();

  // Non-owning of the edge edge_weights object
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  //
  // Run example graph algorithms
  //

  run_graph_algorithms<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    *handle, graph_view, edge_weight_view);
}

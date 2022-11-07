/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

// Author: Xavier Cadet xcadet@nvidia.com

#pragma once
#include <rmm/device_vector.hpp>

namespace cugraph {
namespace detail {
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void betweenness_centrality(raft::handle_t const& handle,
                            legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                            result_t* result,
                            bool normalize,
                            bool endpoints,
                            weight_t const* weight,
                            vertex_t const number_of_sources,
                            vertex_t const* sources);

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void edge_betweenness_centrality(legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                 result_t* result,
                                 bool normalize,
                                 weight_t const* weight,
                                 vertex_t const number_of_sources,
                                 vertex_t const* sources);

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void verify_betweenness_centrality_input(result_t* result,
                                         bool is_edge_betweenness,
                                         bool normalize,
                                         bool endpoints,
                                         weight_t const* weights,
                                         vertex_t const number_of_sources,
                                         vertex_t const* sources);

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
class BC {
 public:
  virtual ~BC(void) {}
  BC(raft::handle_t const& handle,
     legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
     cudaStream_t stream = 0)
    : handle_(handle), graph_(graph)
  {
    setup();
  }
  void configure(result_t* betweenness,
                 bool is_edge_betweenness,
                 bool normalize,
                 bool endpoints,
                 weight_t const* weight,
                 vertex_t const* sources,
                 vertex_t const number_of_sources);

  void configure_edge(result_t* betweenness,
                      bool normalize,
                      weight_t const* weight,
                      vertex_t const* sources,
                      vertex_t const number_of_sources);
  void compute();
  void rescale_by_total_sources_used(vertex_t total_number_of_sources_used);

 private:
  // --- RAFT handle ---
  raft::handle_t const& handle_;
  // --- Information concerning the graph ---
  const legacy::GraphCSRView<vertex_t, edge_t, weight_t>& graph_;
  // --- These information are extracted on setup ---
  vertex_t number_of_vertices_;  // Number of vertices in the graph
  vertex_t number_of_edges_;     // Number of edges in the graph
  edge_t const* offsets_ptr_;    // Pointer to the offsets
  vertex_t const* indices_ptr_;  // Pointers to the indices

  // --- Information from configuration ---
  bool configured_          = false;  // Flag to ensure configuration was called
  bool normalized_          = false;  // If True normalize the betweenness
  bool is_edge_betweenness_ = false;  // If True compute edge_betweeness

  // FIXME: For weighted version
  weight_t const* edge_weights_ptr_ = nullptr;  // Pointer to the weights
  bool endpoints_                   = false;    // If True normalize the betweenness
  vertex_t const* sources_          = nullptr;  // Subset of vertices to gather information from
  vertex_t number_of_sources_;                  // Number of vertices in sources

  // --- Output ----
  // betweenness is set/read by users - using Vectors
  result_t* betweenness_ = nullptr;

  // --- Data required to perform computation ----
  rmm::device_vector<vertex_t> distances_vec_;
  rmm::device_vector<vertex_t> predecessors_vec_;
  rmm::device_vector<double> sp_counters_vec_;
  rmm::device_vector<double> deltas_vec_;

  vertex_t* distances_ =
    nullptr;  // array<vertex_t>(|V|) stores the distances gathered by the latest SSSP
  vertex_t* predecessors_ =
    nullptr;  // array<weight_t>(|V|) stores the predecessors of the latest SSSP
  double* sp_counters_ =
    nullptr;  // array<vertex_t>(|V|) stores the shortest path counter for the latest SSSP
  double* deltas_ = nullptr;  // array<result_t>(|V|) stores the dependencies for the latest SSSP

  int max_grid_dim_1D_  = 0;
  int max_block_dim_1D_ = 0;

  void setup();

  void initialize_work_vectors();
  void initialize_pointers_to_vectors();
  void initialize_device_information();

  void compute_single_source(vertex_t source_vertex);

  void accumulate(vertex_t source_vertex, vertex_t max_depth);
  void initialize_dependencies();
  void accumulate_edges(vertex_t max_depth, dim3 grid_configuration, dim3 block_configuration);
  void accumulate_vertices_with_endpoints(vertex_t source_vertex,
                                          vertex_t max_depth,
                                          dim3 grid_configuration,
                                          dim3 block_configuration);
  void accumulate_vertices(vertex_t max_depth, dim3 grid_configuration, dim3 block_configuration);
  void add_reached_endpoints_to_source_betweenness(vertex_t source_vertex);
  void add_vertices_dependencies_to_betweenness();

  void rescale();
  std::tuple<result_t, bool> rescale_vertices_betweenness_centrality(result_t rescale_factor,
                                                                     bool modified);
  std::tuple<result_t, bool> rescale_edges_betweenness_centrality(result_t rescale_factor,
                                                                  bool modified);
  void apply_rescale_factor_to_betweenness(result_t scaling_factor);
};
}  // namespace detail
}  // namespace cugraph

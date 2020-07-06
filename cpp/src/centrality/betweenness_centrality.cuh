/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <rmm/thrust_rmm_allocator.h>

namespace cugraph {
namespace detail {
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(GraphCSRView<VT, ET, WT> const &graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT const number_of_sources,
                            VT const *sources);

template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality(GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalize,
                                 WT const *weight,
                                 VT const number_of_sources,
                                 VT const *sources);

template <typename VT, typename ET, typename WT, typename result_t>
void verify_betweenness_centrality_input(result_t *result,
                                         bool is_edge_betweenness,
                                         bool normalize,
                                         bool endpoints,
                                         WT const *weights,
                                         VT const number_of_sources,
                                         VT const *sources);

template <typename VT, typename ET, typename WT, typename result_t>
class BC {
 public:
  virtual ~BC(void) {}
  BC(GraphCSRView<VT, ET, WT> const &graph, cudaStream_t stream = 0)
    : graph_(graph), stream_(stream)
  {
    setup();
  }
  void configure(result_t *betweenness,
                 bool is_edge_betweenness,
                 bool normalize,
                 bool endpoints,
                 WT const *weight,
                 VT const *sources,
                 VT const number_of_sources);

  void configure_edge(result_t *betweenness,
                      bool normalize,
                      WT const *weight,
                      VT const *sources,
                      VT const number_of_sources);
  void compute();

 private:
  // --- Information concerning the graph ---
  const GraphCSRView<VT, ET, WT> &graph_;
  // --- These information are extracted on setup ---
  VT number_of_vertices_;  // Number of vertices in the graph
  VT number_of_edges_;     // Number of edges in the graph
  ET const *offsets_ptr_;  // Pointer to the offsets
  VT const *indices_ptr_;  // Pointers to the indices

  // --- Information from configuration ---
  bool configured_          = false;  // Flag to ensure configuration was called
  bool normalized_          = false;  // If True normalize the betweenness
  bool is_edge_betweenness_ = false;  // If True compute edge_betweeness

  // FIXME: For weighted version
  WT const *edge_weights_ptr_ = nullptr;  // Pointer to the weights
  bool endpoints_             = false;    // If True normalize the betweenness
  VT const *sources_          = nullptr;  // Subset of vertices to gather information from
  VT number_of_sources_;                  // Number of vertices in sources

  // --- Output ----
  // betweenness is set/read by users - using Vectors
  result_t *betweenness_ = nullptr;

  // --- Data required to perform computation ----
  rmm::device_vector<VT> distances_vec_;
  rmm::device_vector<VT> predecessors_vec_;
  rmm::device_vector<double> sp_counters_vec_;
  rmm::device_vector<double> deltas_vec_;

  VT *distances_    = nullptr;  // array<VT>(|V|) stores the distances gathered by the latest SSSP
  VT *predecessors_ = nullptr;  // array<WT>(|V|) stores the predecessors of the latest SSSP
  double *sp_counters_ =
    nullptr;                  // array<VT>(|V|) stores the shortest path counter for the latest SSSP
  double *deltas_ = nullptr;  // array<result_t>(|V|) stores the dependencies for the latest SSSP

  // FIXME: This should be replaced using RAFT handle
  int device_id_        = 0;
  int max_grid_dim_1D_  = 0;
  int max_block_dim_1D_ = 0;
  cudaStream_t stream_;

  void setup();

  void initialize_work_vectors();
  void initialize_pointers_to_vectors();
  void initialize_device_information();

  void compute_single_source(VT source_vertex);

  void accumulate(VT source_vertex, VT max_depth);
  void initialize_dependencies();
  void accumulate_edges(VT max_depth, dim3 grid_configuration, dim3 block_configuration);
  void accumulate_vertices_with_endpoints(VT source_vertex,
                                          VT max_depth,
                                          dim3 grid_configuration,
                                          dim3 block_configuration);
  void accumulate_vertices(VT max_depth, dim3 grid_configuration, dim3 block_configuration);
  void add_reached_endpoints_to_source_betweenness(VT source_vertex);
  void add_vertices_dependencies_to_betweenness();

  void rescale();
  void rescale_vertices_betweenness_centrality(result_t &rescale_factor, bool &modified);
  void rescale_edges_betweenness_centrality(result_t &rescale_factor, bool &modified);
  void apply_rescale_factor_to_betweenness(result_t scaling_factor);
};
}  // namespace detail
}  // namespace cugraph

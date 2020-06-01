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
class BC {
 public:
  virtual ~BC(void) {}
  BC(experimental::GraphCSRView<VT, ET, WT> const &_graph, cudaStream_t _stream = 0)
    : graph(_graph), stream(_stream)
  {
    setup();
  }
  void configure(result_t *betweenness,
                 bool is_edge_betweenness,
                 bool normalize,
                 bool endpoints,
                 WT const *weigth,
                 VT const *sources,
                 VT const number_of_sources);
  // TODO(xcadet) This should probably be merged in a single function
  void configure_edge(result_t *betweenness,
                      bool normalize,
                      WT const *weigth,
                      VT const *sources,
                      VT const number_of_sources);
  void compute();
  // void compute_edge();

 private:
  // --- Information concerning the graph ---
  const experimental::GraphCSRView<VT, ET, WT> &graph;
  // --- These information are extracted on setup ---
  VT number_of_vertices;  // Number of vertices in the graph
  VT number_of_edges;     // Number of edges in the graph
  ET const *offsets_ptr;  // Pointer to the offsets
  VT const *indices_ptr;  // Pointers to the indices

  // --- Information from configuration ---
  bool configured          = false;  // Flag to ensure configuration was called
  bool normalized          = false;  // If True normalize the betweenness
  bool is_edge_betweenness = false;  // If True compute edge_betweeness

  // FIXME: For weighted version
  WT const *edge_weights_ptr = nullptr;  // Pointer to the weights
  bool endpoints             = false;    // If True normalize the betweenness
  VT const *sources          = nullptr;  // Subset of vertices to gather information from
  VT number_of_sources;                  // Number of vertices in sources

  // --- Output ----
  // betweenness is set/read by users - using Vectors
  result_t *betweenness = nullptr;

  // --- Data required to perform computation ----
  rmm::device_vector<VT> distances_vec;
  rmm::device_vector<VT> predecessors_vec;
  rmm::device_vector<double> sp_counters_vec;
  rmm::device_vector<double> deltas_vec;

  VT *distances    = nullptr;  // array<VT>(|V|) stores the distances gathered by the latest SSSP
  VT *predecessors = nullptr;  // array<WT>(|V|) stores the predecessors of the latest SSSP
  double *sp_counters =
    nullptr;                 // array<VT>(|V|) stores the shortest path counter for the latest SSSP
  double *deltas = nullptr;  // array<result_t>(|V|) stores the dependencies for the latest SSSP

  // FIXME: This should be replaced using RAFT handle
  int device_id        = 0;
  int max_grid_dim_1D  = 0;
  int max_block_dim_1D = 0;
  cudaStream_t stream;

  // -----------------------------------------------------------------------
  void setup();  // Saves information related to the graph itself

  void accumulate(result_t *betweenness,
                  VT *distances,
                  double *sp_counters,
                  double *deltas,
                  VT source,
                  VT max_depth);

  void accumulate_edges(result_t *betweenness,
                        VT *distances,
                        double *sp_counters,
                        double *deltas,
                        VT source,
                        VT max_depth);

  void accumulate_endpoints(result_t *betweenness,
                            VT *distances,
                            double *sp_counters,
                            double *deltas,
                            VT source,
                            VT max_depth);

  void compute_single_source(VT source_vertex);

  void initialize_work_sizes();
  void initialize_pointers_to_vectors();
  void initialize_device_information();

  void rescale();
  void rescale_vertices_betweenness_centrality(result_t &rescale_factor,
                                               bool endpoints,
                                               bool &modified);
  void rescale_edges_betweenness_centrality(result_t &rescale_factor, bool &modified);
};
}  // namespace detail
}  // namespace cugraph

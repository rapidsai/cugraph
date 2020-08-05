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
#pragma once

#include <rmm/thrust_rmm_allocator.h>

#include <graph.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t modularity(weight_t total_edge_weight,
                    weight_t resolution,
                    GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                    vertex_t const *d_cluster,
                    cudaStream_t stream = 0);

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_delta_modularity(
  weight_t total_edge_weight,
  weight_t resolution,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<vertex_t> const &src_indices_v,
  rmm::device_vector<weight_t> const &vertex_weights_v,
  rmm::device_vector<weight_t> const &cluster_weights_v,
  rmm::device_vector<vertex_t> const &cluster_v,
  rmm::device_vector<vertex_t> &cluster_hash_v,
  rmm::device_vector<weight_t> &delta_Q_v,
  rmm::device_vector<weight_t> &tmp_size_V_v,
  cudaStream_t stream = 0);

template <typename vertex_t, typename edge_t, typename weight_t>
void louvain(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
             weight_t *final_modularity,
             int *num_level,
             vertex_t *cluster_vec,
             int max_level,
             weight_t resolution,
             cudaStream_t stream = 0);

}  // namespace detail
}  // namespace cugraph

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

#include <graph.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void louvain(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
             weight_t *final_modularity,
             int *num_level,
             vertex_t *cluster_vec,
             int max_level,
             cudaStream_t stream = 0);

}  // namespace detail
}  // namespace cugraph

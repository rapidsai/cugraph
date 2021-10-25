/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/cugraph_api.h>

#ifdef __cplusplus
extern "C" {
#endif

cugraph_error_t pagerank(const cugraph_raft_handle_t* handle,
                         const cugraph_graph_t* graph,
                         cugraph_type_erased_device_array_t* pageranks,
                         cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
                         double alpha,
                         double epsilon,
                         size_t max_iterations,
                         bool has_initial_guess,
                         bool do_expensive_check);

cugraph_error_t personalized_pagerank(
  const cugraph_raft_handle_t* handle,
  const cugraph_graph_t* graph,
  cugraph_type_erased_device_array_t* pageranks,
  cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
  cugraph_type_erased_device_array_t* personalization_vertices,
  cugraph_type_erased_device_array_t* personalization_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool do_expensive_check);

cugraph_error_t bfs(const cugraph_raft_handle_t* handle,
                    const cugraph_graph_t* graph,
                    cugraph_type_erased_device_array_t* distances,
                    cugraph_type_erased_device_array_t* predecessors,
                    const cugraph_type_erased_device_array_t* sources,
                    bool direction_optimizing,
                    vertex_t depth_limit,
                    bool do_expensive_check);

#ifdef __cplusplus
}
#endif

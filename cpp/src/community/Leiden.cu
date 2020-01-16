/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cugraph.h>
#include "utilities/error_utils.h"
#include <rmm_utils.h>
#include "utilities/graph_utils.cuh"

namespace cugraph {
template<typename IdxT, typename ValT>
void leiden(Graph* graph,
            int metric,
            double gamma,
            IdxT* leiden_parts,
            int max_iter = 100){
  // Check for error conditions
  CUGRAPH_EXPECTS(graph != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(leiden_parts != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Graph must have adjacency list");
  CUGRAPH_EXPECTS(graph->adjList->edge_data != nullptr, "Graph must have weights");

  // Get info about the graph
  IdxT n = graph->adjList->offsets->size - 1;
  IdxT nnz = graph->adjList->indices->size;

  // Assign initial singleton partition
  thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr), leiden_parts, leiden_parts + n, 0);

  // Compute metric

  // Compute delta metric

  // Reassign nodes with positive delta metric

  // Repeat until no swaps are made

  // Refine the partition

  // Aggregate the graph according to the refined partition

  // Set initial partitioning according to unrefined partition

  //

}

// Explicit template instantiations
template void leiden<int32_t, float>(Graph* graph,
                                     int metric,
                                     double gamma,
                                     int32_t* leiden_parts,
                                     int max_iter);
template void leiden<int32_t, double>(Graph* graph,
                                      int metric,
                                      double gamma,
                                      int32_t* leiden_parts,
                                      int max_iter);
template void leiden<int64_t, float>(Graph* graph,
                                     int metric,
                                     double gamma,
                                     int64_t* leiden_parts,
                                     int max_iter);
template void leiden<int64_t, double>(Graph* graph,
                                      int metric,
                                      double gamma,
                                      int64_t* leiden_parts,
                                      int max_iter);

} // cugraph namespace

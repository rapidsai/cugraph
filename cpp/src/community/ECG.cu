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
/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_gdf.cu
 * ---------------------------------------------------------------------------**/

#include <cugraph.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include <ctime>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"
#include <rmm_utils.h>

namespace {
template <typename IdxT, typename ValT>
void ecg_impl(cugraph::Graph* graph,
              double min_weight,
              int ensemble_size,
              gdf_column *ecg_parts) {
  for (int i = 0; i < ensemble_size; i++) {
    // Take ensemble_size random permutations of the graph and run Louvain clustering on each

    // For each edge in the graph determine whether the endpoints are in the same partition

    // Keep a sum for each edge of the total number of times its endpoints are in the same partition
  }

  // Set weights = min_weight + (1 - min-weight)*sum/ensemble_size

  // Run Louvain on the original graph using the computed weights
}
} // anonymous namespace


namespace cugraph {
void ecg(Graph* graph,
         double min_weight,
         int ensemble_size,
         gdf_column *ecg_parts) {
  CUGRAPH_EXPECTS(graph != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(ecg_parts != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Graph must have adjacency list");

  // determine the index type and value type of the graph
  // Call the appropriate templated instance of the implementation
  switch (graph->adjList->offsets->dtype) {
    case GDF_INT32: {
      switch (graph->adjList->edge_data) {
        case GDF_FLOAT32: {
          ecg_impl<int32_t, float>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
        case GDF_FLOAT64: {
          ecg_impl<int32_t, double>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
      }
      break;
    }
    case GDF_INT64: {
      switch (graph->adjList->edge_data) {
        case GDF_FLOAT32: {
          ecg_impl<int64_t, float>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
        case GDF_FLOAT64: {
          ecg_impl<int64_t, double>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
      }
    }
  }
}
} // cugraph namespace

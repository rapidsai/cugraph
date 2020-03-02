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

// Author: Xavier Cadet xcadet@nvidia.com
#include <cugraph.h>
#include <rmm_utils.h>

#include "betweenness_centrality.cuh"

namespace cugraph {
namespace detail {

template <typename VT, typename WT>
void BC<VT, WT>::setup() {
}

template <typename VT, typename WT>
void BC<VT, WT>::traverse() {
    // We need sigmas / deltas
    // for source in nodes
    //for (auto source : graph->edge_data) {
        //std:cout << source << "\n"
    //}
}

template <typename IndexType, typename BetweennessType>
void BC<IndexType, BetweennessType>::clean() {
}

//  --
template <typename VT, typename ET, typename WT>
void _check_input(Graph *graph, WT *betweenness) {
}

} //namespace detail
  /**
  * ---------------------------------------------------------------------------*
  * @brief Native betweenness centrality
  *
  * @file betweenness_centrality.cu
  * --------------------------------------------------------------------------*/
  template <typename VT, typename ET, typename WT>
  void betweenness_centrality(Graph *graph, WT *betweenness) {
    CUGRAPH_EXPECTS(graph->edgeList != nullptr, "Invalid API parameter: graph edgeList is NULL");
    CUGRAPH_EXPECTS(betweenness != nullptr, "Invalid API parameter: output is nullptr");

    if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double)) {
        CUGRAPH_FAIL("Unsupported betweenness data type, please use float or double");
    }
    // TODO fix me after gdf_column is removed from Graph
    CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32,
                    "Unsupported data type");
    CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32,
                    "Unsupported data type");
    // Handle Unweighted
    if (!graph->edgeList->edge_data) {
    } else { // Handle weighted graph
        CUGRAPH_EXPECTS(
            graph->edgeList->edge_data->size == graph->edgeList->src_indices->size,
            "Graph sizes mismatch");
        // TODO fix me after gdf_column is removed from Graph
        CUGRAPH_EXPECTS(graph->edgeList->edge_data->dtype == GDF_FLOAT32 ||
                        graph->edgeList->edge_data->dtype == GDF_FLOAT64,
                    "Invalid API parameter");
        // TODO fix me after gdf_column is removed from Graph
        // if (distances) CUGRAPH_EXPECTS(typeid(distances) == typeid(graph->adjList->edge_data), "distances and weights type mismatch");

        // BC relies on SSSP which is not defined for graphs with negative weight cycles
        // Warn user about any negative edges
        if (graph->prop && graph->prop->has_negative_edges == GDF_PROP_TRUE)
        std::cerr << "WARN: The graph has negative weight edges. BC will not "
                    "converge if the graph has negative weight cycles\n";
    }
    // Verify that WT is either float or double
    if (typeid(WT) == typeid(float) || typeid(WT) == typeid(double)) {
        cugraph::detail::BC<VT, WT> bc(graph, betweenness);
        //bc.configure(distances, predecessors, sp_counters, nullptr);
        //sssp.traverse(source_vertex);
    } else { // Otherwise the datatype is invalid
        CUGRAPH_EXPECTS(graph->edgeList->edge_data->dtype == GDF_FLOAT32 ||
                        graph->edgeList->edge_data->dtype == GDF_FLOAT64,
                        "Invalid API parameter");
    }
  }

  // explicit instantiation
  template void betweenness_centrality<int, int, float>(Graph *graph, float *betweenness);
  template void betweenness_centrality<int, int, double>(Graph *graph, double *betweenness);
} //namespace cugraph
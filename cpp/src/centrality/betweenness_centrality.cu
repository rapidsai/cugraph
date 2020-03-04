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
    // --- Set up parameters from graph adjList ---
    number_vertices  = graph-> adjList->offsets->size - 1;
    //number_vertices  = graph-> adjList->offsets->size;
    number_edges = graph->adjList->indices->size;
    offsets_ptr = (int*)graph->adjList->offsets->data;
    indices_ptr = (int*)graph->adjList->indices->data;

    edge_weights_ptr = static_cast<WT*>(graph->adjList->edge_data->data);
}

template <typename VT, typename WT>
void BC<VT, WT>::configure(WT *_betweenness) {
    // --- Working data allocation ---
    /*
    thrust::device_vector<VT> predecessors(number_vertices);
    thrust::device_vector<VT> sp_counters(sp_counters);

    thrust::device_vector<WT> sigmas(number_vertices);
    thrust::device_vector<WT> deltas(number_vertices);
    */
    /*
    ALLOC_TRY(&predecessors, sizeof(VT) * number_vertices, nullptr);
    ALLOC_TRY(&sp_counters, sizeof(VT) * number_vertices, nullptr);

    ALLOC_TRY(&sigmas, sizeof(WT) * number_vertices, nullptr);
    ALLOC_TRY(&deltas, sizeof(WT) * number_vertices, nullptr);
    */

    //VT *ptr = thrust::raw_pointer_cast(&predecessors[0]);
    //std::cout << "Predecessors is nullptr " << (ptr == nullptr) << "\n";
    // --- Bind betweenness output vector to internal ---
    betweenness = _betweenness;
    // --- Confirm that configuration went through ---
    configured = true;
}

template <typename VT, typename WT>
void BC<VT, WT>::compute() {
    CUGRAPH_EXPECTS(configured, "BC must be configured before computation");
    std::cout << "There are " << number_vertices << " nodes\n";

    thrust::device_vector<VT> d_predecessors(number_vertices, static_cast<VT>(0));
    thrust::device_vector<VT> d_sp_counters(number_vertices, static_cast<VT>(0));

    //thrust::device_vector<WT> d_betweenness(number_vertices, static_cast<WT>(0));

    thrust::host_vector<VT> h_predecessors(number_vertices, static_cast<VT>(0));
    thrust::host_vector<VT> h_sp_counters(number_vertices, static_cast<VT>(0));
    thrust::host_vector<WT> h_betweenness(number_vertices, static_cast<WT>(0));


    VT *d_predecessors_ptr = thrust::raw_pointer_cast(&d_predecessors[0]);
    VT *d_sp_counters_ptr = thrust::raw_pointer_cast(&d_sp_counters[0]);

    for (int source_vertex = 0; source_vertex < number_vertices ; ++source_vertex) {
        // Step 1) Singe-source shortest-path problem
        cugraph::sssp(graph, static_cast<WT *>(nullptr), d_predecessors_ptr, d_sp_counters_ptr, source_vertex);
        thrust::copy(d_predecessors.begin(), d_predecessors.end(), h_predecessors.begin());
        thrust::copy(d_sp_counters.begin(), d_sp_counters.end(), h_sp_counters.begin());
        cudaDeviceSynchronize();
        // TODO(xcadet) Remove printing information
        std::cout << "Sigmas for source " << source_vertex << ": ";
        thrust::copy(h_sp_counters.begin(), h_sp_counters.end(), std::ostream_iterator<float>(std::cout, ", "));
        std::cout << "\n";
        // Step 2) Path reconstruction or S equivalent generation?

        //cudaDeviceSynchronize();
        // sp_counters should be in floating points for the next phase
        // Step 3) Accumulation
        accumulate(h_betweenness, h_predecessors, h_sp_counters, source_vertex);
    }
    std::cout << "Betweeness: ";
    thrust::copy(h_betweenness.begin(), h_betweenness.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "\n";
}


template <typename VT, typename WT>
void BC<VT, WT>::accumulate(thrust::host_vector<WT> &h_betweenness,
                            thrust::host_vector<VT> &h_predecessors,
                            thrust::host_vector<VT> &h_sp_counters,
                            VT source) {
    // TODO(xcadet) Remove the debugs messages
    std::cout << "[CUDA] Accumulating from " << source << "\n";
    std::cout << "\tPredecessors: ";
    thrust::copy(h_predecessors.begin(), h_predecessors.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "\n";
    std::cout << "sp_counters: ";
    thrust::copy(h_sp_counters.begin(), h_sp_counters.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "\n";

    thrust::host_vector<WT> h_deltas(number_vertices, static_cast<WT>(0));
    // TODO(xcadet) There is most likely a more efficient way to handle it
    for (VT w = 0; w < number_vertices; ++w) {
        WT factor = (static_cast<WT>(1.0) + h_deltas[w]) / static_cast<WT>(h_sp_counters[w]);
        // TODO(xcadet) The current SSSP implementation only stores 1 Node
        VT v = h_predecessors[w];
        if (v != -1) { // This node has predecessor
            h_deltas[v] = h_deltas[v] + static_cast<WT>(h_sp_counters[v]) * factor;
            std::cout << "Updated depencies for node " << v << " with " << h_deltas[v] << "\n";
        } // We should not updated our dependencies
        // The node is different than the source
        if (w != source) {
            h_betweenness[w] += h_deltas[w];
            std::cout << "Betweenness for " << w << " updated to " << h_betweenness[w] << "\n";
        }
    }

}

template <typename IndexType, typename BetweennessType>
void BC<IndexType, BetweennessType>::clean() {
    //ALLOC_FREE_TRY(predecessors, nullptr);
    //ALLOC_FREE_TRY(sp_counters, nullptr);
    //ALLOC_FREE_TRY(sigmas, nullptr);
    //ALLOC_FREE_TRY(deltas, nullptr);
    // ---  Betweenness is not ours ---
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
    CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter: graph adjList is NULL");
    CUGRAPH_EXPECTS(betweenness != nullptr, "Invalid API parameter: output is nullptr");

    if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double)) {
        CUGRAPH_FAIL("Unsupported betweenness data type, please use float or double");
    }
    // TODO fix me after gdf_column is removed from Graph
    CUGRAPH_EXPECTS(graph->adjList->offsets->dtype == GDF_INT32,
                    "Unsupported data type");
    CUGRAPH_EXPECTS(graph->adjList->indices->dtype == GDF_INT32,
                    "Unsupported data type");
    // Handle Unweighted
    if (!graph->adjList->edge_data) {

    } else { // Handle weighted graph
        CUGRAPH_EXPECTS(
            graph->adjList->edge_data->size == graph->adjList->indices->size,
            "Graph sizes mismatch");
        // TODO fix me after gdf_column is removed from Graph
        CUGRAPH_EXPECTS(graph->adjList->edge_data->dtype == GDF_FLOAT32 ||
                        graph->adjList->edge_data->dtype == GDF_FLOAT64,
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
        cugraph::detail::BC<VT, WT> bc(graph);
        bc.configure(betweenness);
        bc.compute();
    } else { // Otherwise the datatype is invalid
        CUGRAPH_EXPECTS(graph->adjList->edge_data->dtype == GDF_FLOAT32 ||
                        graph->adjList->edge_data->dtype == GDF_FLOAT64,
                        "Invalid API parameter");
    }
  }

  // explicit instantiation
  template void betweenness_centrality<int, int, float>(Graph *graph, float *betweenness);
  template void betweenness_centrality<int, int, double>(Graph *graph, double *betweenness);
} //namespace cugraph
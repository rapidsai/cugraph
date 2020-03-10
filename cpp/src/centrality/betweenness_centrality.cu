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

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::setup() {
    // --- Set up parameters from graph adjList ---
    number_vertices  = graph-> adjList->offsets->size - 1;

    number_edges = graph->adjList->indices->size;
    offsets_ptr = (int*)graph->adjList->offsets->data;
    indices_ptr = (int*)graph->adjList->indices->data;

    edge_weights_ptr = static_cast<WT*>(graph->adjList->edge_data->data);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::configure(result_t *_betweenness, bool _normalize,
                                         VT const *_sample_seeds,
                                         VT _number_of_sample_seeds) {
    // --- Working data allocation ---
    // --- Bind betweenness output vector to internal ---
    betweenness = _betweenness;
    apply_normalization = _normalize;
    sample_seeds = _sample_seeds;
    number_of_sample_seeds =  _number_of_sample_seeds;
    // --- Confirm that configuration went through ---
    configured = true;
}

template <typename T>
struct ifNegativeReplace {
  __host__ __device__
  T operator()(const T& dist, const T& node) const
  {
    return (dist == -1) ? -1 : node;
  }
};

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::normalize() {
    thrust::device_vector<result_t> normalizer(number_vertices);
    thrust::fill(normalizer.begin(), normalizer.end(), ((number_vertices - 1) * (number_vertices - 2)));

    if (typeid(result_t) == typeid(float)) {
        thrust::transform(rmm::exec_policy(stream)->on(stream), betweenness, betweenness + number_vertices, normalizer.begin(), betweenness, thrust::divides<float>());
    } else if (typeid(result_t) == typeid(double)) {
        thrust::transform(rmm::exec_policy(stream)->on(stream), betweenness, betweenness + number_vertices, normalizer.begin(), betweenness, thrust::divides<double>());
    }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate(thrust::host_vector<result_t> &h_betweenness,
                            thrust::host_vector<VT> &h_nodes,
                            thrust::host_vector<VT> &h_predecessors,
                            thrust::host_vector<VT> &h_sp_counters,
                            VT source) {
    // TODO(xcadet) Remove the debugs messages (+ 1 are for testing on line3-False-1.0 against Python custom test)
    /*
    std::cout << "[CUDA] Accumulating from " << source << "\n";
    std::cout << "\t[CUDA] Predecessors: ";
    thrust::copy(h_predecessors.begin(), h_predecessors.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "\n";
    std::cout << "\t[CUDA]sp_counters: ";
    thrust::copy(h_sp_counters.begin(), h_sp_counters.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "\n";
    */

    thrust::host_vector<result_t> h_deltas(number_vertices, static_cast<WT>(0));
    // TODO(xcadet) There is most likely a more efficient way to handle it
    for (VT w : h_nodes) {
        if (w == -1) { // The nodes after this ones have not been visited and should not update anything
            break;
        }
        //std::cout << "\t[CUDA] Visiting " << w << "\n";
        result_t factor = (static_cast<result_t>(1.0) + h_deltas[w]) / static_cast<result_t>(h_sp_counters[w]);
        // TODO(xcadet) The current SSSP implementation only stores 1 Node
        VT v = h_predecessors[w];
        if (v != -1) { // This node has predecessor
            WT old = h_deltas[v];
            h_deltas[v] += static_cast<result_t>(h_sp_counters[v]) * factor;
            //std::cout << "\t\t[CUDA] Updated depencies for node " << v << " with " << h_deltas[v] << "\n";
            //std::cout << "\t\t\t[CUDA] From " << old << " to " << h_deltas[v] << ", with factor = " << factor << "\n";
        } // We should not updated our dependencies
        // The node is different than the source
        if (w != source) {
            h_betweenness[w] += h_deltas[w];
            //std::cout << "\t\t[CUDA] Betweenness for " << w << " updated to " << h_betweenness[w] << "\n";
        }
    }

}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::clean() {
    //ALLOC_FREE_TRY(predecessors, nullptr);
    //ALLOC_FREE_TRY(sp_counters, nullptr);
    //ALLOC_FREE_TRY(sigmas, nullptr);
    //ALLOC_FREE_TRY(deltas, nullptr);
    // ---  Betweenness is not ours ---
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::check_input() {
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute() {
    CUGRAPH_EXPECTS(configured, "BC must be configured before computation");

    // TODO(xcadet) There are too many of them
    thrust::device_vector<WT> d_distances(number_vertices, static_cast<WT>(0));
    thrust::device_vector<VT> d_predecessors(number_vertices, static_cast<VT>(0));
    thrust::device_vector<VT> d_sp_counters(number_vertices, static_cast<VT>(0));

    thrust::host_vector<WT> h_distances(number_vertices, static_cast<WT>(0));
    thrust::host_vector<VT> h_predecessors(number_vertices, static_cast<VT>(0));
    thrust::host_vector<VT> h_sp_counters(number_vertices, static_cast<VT>(0));
    thrust::host_vector<result_t> h_betweenness(number_vertices, static_cast<result_t>(0));

    thrust::host_vector<VT> h_nodes(number_vertices);

    WT *d_distances_ptr = thrust::raw_pointer_cast(&d_distances[0]);
    VT *d_predecessors_ptr = thrust::raw_pointer_cast(&d_predecessors[0]);
    VT *d_sp_counters_ptr = thrust::raw_pointer_cast(&d_sp_counters[0]);

    for (int source_vertex = 0; source_vertex < number_vertices ; ++source_vertex) {
        // Step 0) Set information for upcoming sssp
        thrust::sequence(thrust::host, h_nodes.begin(), h_nodes.end(), 0);

        // Step 1) Singe-source shortest-path problem
        cugraph::sssp(graph, d_distances_ptr, d_predecessors_ptr, d_sp_counters_ptr, source_vertex);
        thrust::copy(d_distances.begin(), d_distances.end(), h_distances.begin());
        thrust::copy(d_predecessors.begin(), d_predecessors.end(), h_predecessors.begin());
        thrust::copy(d_sp_counters.begin(), d_sp_counters.end(), h_sp_counters.begin());

        // Step 2) We need to generate the nodes order (leverage distance to keep sssp lighter) ?
        thrust::replace(thrust::host, h_distances.begin(), h_distances.end(), std::numeric_limits<WT>::max(), static_cast<WT>(-1));
        thrust::sort_by_key(thrust::host, h_distances.begin(), h_distances.end(), h_nodes.begin(), thrust::greater<VT>());
        thrust::transform(thrust::host, h_distances.begin(), h_distances.end(), h_nodes.begin(), h_nodes.begin(), ifNegativeReplace<WT>());
        cudaDeviceSynchronize(); // TODO(xcadet) Is this one mandatory ?
        // Step 3) Accumulation
        accumulate(h_betweenness, h_nodes, h_predecessors, h_sp_counters, source_vertex);
        /*
        std::cout << "Betweeness fter " << source_vertex << "\n";
        thrust::copy(h_betweenness.begin(), h_betweenness.end(), std::ostream_iterator<float>(std::cout, ", "));
        std::cout << "\n";
        */
        //break;

    }
    // Step 4: Rescale results based on number of vertices and directed or u
    cudaMemcpyAsync(betweenness, &h_betweenness[0],
                    number_vertices * sizeof(WT),
                    cudaMemcpyHostToDevice, stream);
    cudaDeviceSynchronize();
    if (apply_normalization) {
        normalize();
    }
}

} //namespace detail
  /**
  * ---------------------------------------------------------------------------*
  * @brief Native betweenness centrality
  *
  * @file betweenness_centrality.cu
  * --------------------------------------------------------------------------*/
  template <typename VT, typename ET, typename WT, typename result_t>
  //void betweenness_centrality(Graph *graph, WT *betweenness, bool normalized, w) {
  // TODO(xcadet) use experimental::GraphCSR<VT,ET,WT> const &graph
  void betweenness_centrality(Graph *graph, result_t *betweenness, bool normalize,
                              VT const *sample_seeds,
                              VT number_of_sample_seeds) {
    CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter: graph adjList is NULL");
    CUGRAPH_EXPECTS(betweenness != nullptr, "Invalid API parameter: output is nullptr");

    if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double)) {
        CUGRAPH_FAIL("Unsupported betweenness data type, please use float or double");
    }

    CUGRAPH_EXPECTS(sample_seeds == nullptr, "Sampling seeds is currently not supported");
    // TODO fix me after gdf_column is removed from Graph
    CUGRAPH_EXPECTS(graph->adjList->offsets->dtype == GDF_INT32,
                    "Unsupported data type");
    CUGRAPH_EXPECTS(graph->adjList->indices->dtype == GDF_INT32,
                    "Unsupported data type");

    // Handle Unweighted
    if (!graph->adjList->edge_data) {
        // Generate unit weights

        void* d_edge_data;
        graph->adjList->edge_data = new gdf_column;
        cudaStream_t stream{nullptr};

        std::vector<WT> h_edge_data(graph->adjList->indices->size, 1.0);
        size_t edge_data_size = sizeof(WT) * h_edge_data.size();
        ALLOC_TRY((void**)&d_edge_data, edge_data_size, stream);
        CUDA_TRY(cudaMemcpy(d_edge_data,
                            &h_edge_data[0],
                            edge_data_size,
                            cudaMemcpyHostToDevice));
        gdf_column_view(graph->adjList->edge_data,
                        d_edge_data,
                        nullptr,
                        graph->adjList->indices->size,
                        (typeid(WT) == typeid(double)) ? GDF_FLOAT64 : GDF_FLOAT32);
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
        cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
        bc.configure(betweenness, normalize, sample_seeds, number_of_sample_seeds);
        bc.compute();
    } else { // Otherwise the datatype is invalid
        CUGRAPH_EXPECTS(graph->adjList->edge_data->dtype == GDF_FLOAT32 ||
                        graph->adjList->edge_data->dtype == GDF_FLOAT64,
                        "Invalid API parameter");
    }
  }

  // explicit instantiation
  template void betweenness_centrality<int, int, float, float>(Graph *graph, float *betweenness, bool, int const *, int);
  template void betweenness_centrality<int, int, double, double>(Graph *graph, double *betweenness, bool, int const *, int);
} //namespace cugraph
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
    // --- Bind betweenness output vector to internal ---
    betweenness = _betweenness;
    apply_normalization = _normalize;
    sample_seeds = _sample_seeds;
    number_of_sample_seeds =  _number_of_sample_seeds;

    // --- Working data allocation ---
    ALLOC_TRY(&distances, number_vertices * sizeof(WT), nullptr);
    ALLOC_TRY(&predecessors, number_vertices * sizeof(VT), nullptr);
    ALLOC_TRY(&nodes, number_vertices * sizeof(VT), nullptr);
    ALLOC_TRY(&sp_counters, number_vertices * sizeof(int), nullptr);
    ALLOC_TRY(&deltas, number_vertices * sizeof(result_t), nullptr);
    // --- Confirm that configuration went through ---
    configured = true;
}
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::clean() {
    ALLOC_FREE_TRY(distances, nullptr);
    ALLOC_FREE_TRY(predecessors, nullptr);
    ALLOC_FREE_TRY(nodes, nullptr);
    ALLOC_FREE_TRY(sp_counters, nullptr);
    ALLOC_FREE_TRY(deltas, nullptr);
    // ---  Betweenness is not ours ---
}

template <typename WT, typename VT>
struct ifNegativeReplace {
  __host__ __device__
  VT operator()(const WT& dist, const VT& node) const
  {
    return (dist == static_cast<WT>(-1)) ? static_cast<VT>(-1) : node;
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

/* TODO(xcadet) Use an iteration based node system, to process nodes of the same level at the same time
** For now all the work is done on the first thread */
template <typename VT, typename ET, typename WT, typename result_t>
__global__ void accumulate_kernel(result_t *betweenness, VT number_vertices,
                                 VT *nodes, VT *predecessors, int *sp_counters,
                                 result_t *deltas, VT source) {
    int global_id = (blockIdx.x * blockDim.x)  + threadIdx.x;
    if (global_id == 0) { // global_id < number_vertices
        for (int idx = 0; idx < number_vertices; ++idx) {
            VT w = nodes[idx];
            if (w == -1) { // This node and the following have not been visited in the sssp
                break;
            }
            result_t factor = (static_cast<result_t>(1.0) + deltas[w]) / static_cast<result_t>(sp_counters[w]);
            VT v = predecessors[w]; // Multiples nodes could have the same predecessor
            if (v != -1) {
                atomicAdd(&deltas[v], static_cast<result_t>(sp_counters[v]) * factor);
            }
            if (w != source) {
                atomicAdd(&betweenness[w], deltas[w]);
            }
        }
    }
}

// TODO(xcadet) We might be able to handle different nodes with a kernel
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate(result_t *betweenness, VT* nodes,
                                          VT *predecessors, int *sp_counters,
                                          result_t *deltas, VT source) {
    // Step 1) Dependencies (deltas) are initialized to 0 before starting
    thrust::fill(rmm::exec_policy(stream)->on(stream), deltas,
                 deltas + number_vertices, static_cast<result_t>(0));

    // Step 2) Process each node, -1 is used to notify unreached nodes in the sssp
    accumulate_kernel<VT, ET, WT, result_t>
                     <<<1, 1, 0, stream>>>(betweenness, number_vertices,
                                           nodes, predecessors, sp_counters,
                                           deltas, source);
    cudaDeviceSynchronize();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::check_input() {
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute() {
    CUGRAPH_EXPECTS(configured, "BC must be configured before computation");

    for (int source_vertex = 0; source_vertex < number_vertices;
         ++source_vertex) {
        /* Step 0) Set information for upcoming sssp, nodes is reinitialized
        ** as a sequence. [0, number of vertices[ */
        thrust::sequence(rmm::exec_policy(stream)->on(stream), nodes,
                         nodes + number_vertices, 0);

        // Step 1) Singe-source shortest-path problem
        cugraph::sssp(graph, distances, predecessors, sp_counters,
                      source_vertex);

        /* Step 2) To generate the load order (use distances, it is
        ** assumed that the weights are non negative, the accumualtion relies
        ** on this property. */

        /* 2.1: First replace the maximal value of WT by -1 so we can compare
        ** in descending order. */
        thrust::replace(rmm::exec_policy(stream)->on(stream), distances,
                        distances + number_vertices,
                        std::numeric_limits<WT>::max(),
                        static_cast<WT>(-1));

        /* 2.2: Use thrust sort_by_key to sort "nodes" in descending order
        ** based on the distances. */
        thrust::sort_by_key(rmm::exec_policy(stream)->on(stream), distances,
                            distances + number_vertices, nodes,
                            thrust::greater<WT>());

        /* 2.3: -1 in distances meant that the node was never reached,
        ** such vertices are replaced by -1 in "nodes" for the
        accumulation phase. */
        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          distances, distances + number_vertices,
                          nodes, nodes, ifNegativeReplace<WT, VT>());

        cudaDeviceSynchronize(); // TODO(xcadet) Is this mandatory?
        // Step 3) Accumulation
        accumulate(betweenness, nodes, predecessors, sp_counters, deltas, source_vertex);
    }
    // Step 4: Rescale results based on number of vertices and directed or u
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
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

#include <vector>

#include <thrust/transform.h>

#include <algorithms.hpp>
#include <graph.hpp>
#include "rmm_utils.h"

#include <utilities/error_utils.h>

#include <gunrock/gunrock.h>

#include "betweenness_centrality.cuh"

namespace cugraph {

namespace detail {
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::setup() {
    // --- Set up parameters from graph adjList ---
    number_vertices  = graph.number_of_vertices;
    number_edges = graph.number_of_edges;
    offsets_ptr = graph.offsets;
    indices_ptr = graph.indices;
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
        // Step 1) Singe-source shortest-path problem
        cugraph::sssp(graph, distances, predecessors, source_vertex);

        // Step 2) Accumulation
        accumulate(betweenness, nodes, predecessors, sp_counters, deltas, source_vertex);
    }
    cudaDeviceSynchronize();
    if (apply_normalization) {
        normalize();
    }
}
  /**
  * ---------------------------------------------------------------------------*
  * @brief Native betweenness centrality
  *
  * @file betweenness_centrality.cu
  * --------------------------------------------------------------------------*/
  template <typename VT, typename ET, typename WT, typename result_t>
  void betweenness_centrality(experimental::GraphCSR<VT,ET,WT> const &graph,
                            result_t *result,
                            bool normalize,
                            VT const *sample_seeds = nullptr,
                            VT number_of_sample_seeds = 0) {

    CUGRAPH_EXPECTS(result != nullptr, "Invalid API parameter: output betwenness is nullptr");
    if (typeid(VT) != typeid(int)) {
      CUGRAPH_FAIL("Unsupported vertex id data type, please use int");
    }
    if (typeid(ET) != typeid(int)) {
      CUGRAPH_FAIL("Unsupported edge id data type, please use int");
    }
    if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double)) {
      CUGRAPH_FAIL("Unsupported weight data type, please use float or double");
    }

    CUGRAPH_EXPECTS(sample_seeds == nullptr, "Sampling seeds is currently not supported");
    // Current Implementation relies on BFS
    // FIXME: For SSSP version
    // Brandes Algorithm excpets non negative weights for the accumulation
    cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
    bc.configure(result, normalize, sample_seeds, number_of_sample_seeds);
    bc.compute();
  }
} // !cugraph::detail

namespace gunrock {

template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSR<VT,ET,WT> const &graph,
                            result_t *result,
                            bool normalize,
                            VT const *sample_seeds = nullptr,
                            VT number_of_sample_seeds = 0) {

  cudaStream_t stream{nullptr};

  //
  //  gunrock currently (as of 2/28/2020) only operates on a graph and results in
  //  host memory.  [That is, the first step in gunrock is to allocate device memory
  //  and copy the data into device memory, the last step is to allocate host memory
  //  and copy the results into the host memory]
  //
  //  They are working on fixing this.  In the meantime, to get the features into
  //  cuGraph we will first copy the graph back into local memory and when we are finished
  //  copy the result back into device memory.
  //
  std::vector<ET>        v_offsets(graph.number_of_vertices + 1);
  std::vector<VT>        v_indices(graph.number_of_edges);
  std::vector<result_t>  v_result(graph.number_of_vertices);
  std::vector<float>     v_sigmas(graph.number_of_vertices);
  std::vector<int>       v_labels(graph.number_of_vertices);
  
  // fill them
  CUDA_TRY(cudaMemcpy(v_offsets.data(), graph.offsets, sizeof(ET) * (graph.number_of_vertices + 1), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(v_indices.data(), graph.indices, sizeof(VT) * graph.number_of_edges, cudaMemcpyDeviceToHost));

  if (sample_seeds == nullptr) {
    bc(graph.number_of_vertices,
       graph.number_of_edges,
       v_offsets.data(),
       v_indices.data(),
       -1,
       v_result.data(),
       v_sigmas.data(),
       v_labels.data());
  } else {
    //
    //  Gunrock, as currently implemented
    //  doesn't support this method.
    //
    CUGRAPH_FAIL("gunrock doesn't currently support sampling seeds");
  }

  // copy to results
  CUDA_TRY(cudaMemcpy(result, v_result.data(), sizeof(result_t) * graph.number_of_vertices, cudaMemcpyHostToDevice));

  // normalize result
  if (normalize) {
    float denominator = (graph.number_of_vertices - 1) * (graph.number_of_vertices - 2);

    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      result, result + graph.number_of_vertices, result,
                      [denominator] __device__ (float f) {
                        return (f * 2) / denominator;
                      });
  } else {
    //
    //  gunrock answer needs to be doubled to match networkx
    //
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      result, result + graph.number_of_vertices, result,
                      [] __device__ (float f) {
                        return (f * 2);
                      });
  }
}

} // namespace detail

template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSR<VT,ET,WT> const &graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT k,
                            VT const *vertices) {

  //
  // NOTE:  gunrock implementation doesn't yet support the unused parameters:
  //     - endpoints
  //     - weight
  //     - k
  //     - vertices
  //
  // These parameters are present in the API to support future features.
  //
  //gunrock::betweenness_centrality(graph, result, normalize);
  detail::betweenness_centrality(graph, result, normalize);
}

template void betweenness_centrality<int, int, float, float>(experimental::GraphCSR<int,int,float> const &, float*, bool, bool, float const *, int, int const *);

} //namespace cugraph


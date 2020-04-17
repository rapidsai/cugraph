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
    number_of_vertices  = graph.number_of_vertices;
    number_of_edges = graph.number_of_edges;
    offsets_ptr = graph.offsets;
    indices_ptr = graph.indices;
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::configure(result_t *_betweenness, bool _normalize,
                                         VT const *_sources,
                                         VT _number_of_sources) {
    // --- Bind betweenness output vector to internal ---
    betweenness = _betweenness;
    apply_normalization = _normalize;
    sources = _sources;
    number_of_sources = _number_of_sources;

    // --- Working data allocation ---
    ALLOC_TRY(&distances, number_of_vertices * sizeof(VT), nullptr);
    ALLOC_TRY(&predecessors, number_of_vertices * sizeof(VT), nullptr);
    ALLOC_TRY(&sp_counters, number_of_vertices * sizeof(int), nullptr);
    ALLOC_TRY(&deltas, number_of_vertices * sizeof(result_t), nullptr);
    // --- Confirm that configuration went through ---
    configured = true;
}
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::clean() {
    ALLOC_FREE_TRY(distances, nullptr);
    ALLOC_FREE_TRY(predecessors, nullptr);
    ALLOC_FREE_TRY(sp_counters, nullptr);
    ALLOC_FREE_TRY(deltas, nullptr);
    // ---  Betweenness is not ours ---
}

// TODO(xcadet) number_of_sources has to be used for rescale (also add it to reference tests)
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::normalize() {
    printf("[DBG] Being normalized\n");
    thrust::device_vector<result_t> normalizer(number_of_vertices);
    thrust::fill(normalizer.begin(), normalizer.end(), ((number_of_vertices - 1) * (number_of_vertices - 2)));

    if (typeid(result_t) == typeid(float)) {
        thrust::transform(rmm::exec_policy(stream)->on(stream), betweenness, betweenness + number_of_vertices, normalizer.begin(), betweenness, thrust::divides<float>());
    } else if (typeid(result_t) == typeid(double)) {
        thrust::transform(rmm::exec_policy(stream)->on(stream), betweenness, betweenness + number_of_vertices, normalizer.begin(), betweenness, thrust::divides<double>());
    }
}

// Dependecy Accumulation: McLaughlin and Bader, 2018
template <typename VT, typename ET, typename WT, typename result_t>
__global__ void accumulation_kernel(result_t *betweenness, VT number_vertices,
                                  VT const *indices, ET const *offsets,
                                  VT *distances,
                                  int *sp_counters,
                                  result_t *deltas, VT source, VT depth) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < number_vertices;
       tid += gridDim.x * blockDim.x) {
    VT w = tid;
    result_t dsw = 0;
    result_t sw = static_cast<result_t>(sp_counters[w]);
    if (distances[w] == depth) { // Process nodes at this depth
      ET edge_start = offsets[w];
      ET edge_end = offsets[w + 1];
      ET edge_count = edge_end - edge_start;
      for (ET edge_idx = 0; edge_idx < edge_count; ++edge_idx) { // Visit neighbors
        VT v = indices[edge_start + edge_idx];
        if (distances[v] == distances[w] + 1) {
          result_t factor = (static_cast<result_t>(1) + deltas[v]) / static_cast<result_t>(sp_counters[v]);
          dsw += sw * factor;
        }
      }
      deltas[w] = dsw;
    }
  }
}

// TODO(xcadet) We might be able to handle different nodes with a kernel
// With BFS distances can be used to handle accumulation,
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate(result_t *betweenness, VT* distances,
                                          VT *sp_counters,
                                          result_t *deltas, VT source, VT max_depth) {
    dim3 grid, block;
    //block.x = 256; // TODO(xcadet) Replace these values, only for debugging
    block.x = 512;
    grid.x = min(65535, (number_of_edges / block.x + 1));
  // Step 1) Dependencies (deltas) are initialized to 0 before starting
  thrust::fill(rmm::exec_policy(stream)->on(stream), deltas,
               deltas + number_of_vertices, static_cast<result_t>(0));
  // Step 2) Process each node, -1 is used to notify unreached nodes in the sssp
  for (VT depth = max_depth; depth > 0; --depth) {
    //std::cout << "\t[ACC] Processing depth: " << depth << std::endl;
    accumulation_kernel<VT, ET, WT, result_t>
                     <<<grid, block, 0, stream>>>(betweenness, number_of_vertices,
                                             graph.indices, graph.offsets,
                                             distances, sp_counters,
                                             deltas, source, depth);
    cudaDeviceSynchronize();
  }

  thrust::transform(rmm::exec_policy(stream)->on(stream),
    deltas, deltas + number_of_vertices, betweenness, betweenness,
    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::check_input() {
}

// FIXME: Having a system that relies on an class might make it harder to
// dispatch later
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute_single_source(VT source_vertex) {
  std::cout << "[DBG][BC][COMPUTE_SINGLE_SOURCE] Computing from source " << source_vertex << std::endl;
  CUGRAPH_EXPECTS(distances != nullptr, "distances is null");
  CUGRAPH_EXPECTS(predecessors != nullptr, "predecessors is null");
  CUGRAPH_EXPECTS(sp_counters != nullptr, "sp_counters i null");
  // Step 1) Singe-source shortest-path problem
  cugraph::bfs(graph, distances, predecessors, sp_counters, source_vertex,
               graph.prop.directed);
  cudaDeviceSynchronize();

  //TODO(xcadet) Remove that with a BC specific class to gather
  //             information during traversal
  // NOTE: REPLACE INFINITY BY -1 otherwise the max depth will be maximal
  //       value!
  // TODO(xcadet) This could be extracted from the BFS(lvl)
  thrust::replace(rmm::exec_policy(stream)->on(stream), distances,
                  distances + number_of_vertices,
                  std::numeric_limits<VT>::max(),
                  static_cast<VT>(-1));
  auto current_max_depth = thrust::max_element(rmm::exec_policy(stream)->on(stream),
                                               distances,
                                               distances + number_of_vertices);
  VT max_depth = 0;
  cudaMemcpy(&max_depth, current_max_depth, sizeof(VT), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  accumulate(betweenness, distances, sp_counters, deltas, source_vertex, max_depth);
             //*current_max_depth);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute() {
    CUGRAPH_EXPECTS(configured, "BC must be configured before computation");
    // If sources is defined we only process vertices contained in it
    std::cout << "IS SOURCES NUL: " << (sources == nullptr) << std::endl;
    thrust::fill(rmm::exec_policy(stream)->on(stream), betweenness,
                betweenness + number_of_vertices, static_cast<result_t>(0));
    cudaStreamSynchronize(stream);
    if (sources) {
      for (VT source_idx = 0; source_idx < number_of_sources; ++source_idx) {
        VT source_vertex = sources[source_idx];
        compute_single_source(source_vertex);
      }
    } else { // Otherwise process every vertices
      // TODO(xcadet) Maybe we could still use number of sources and set it to number_of_vertices?
      for (VT source_vertex = 0; source_vertex < number_of_vertices;
           ++source_vertex) {
        compute_single_source(source_vertex);
      }
    }
    if (apply_normalization) {
        normalize();
    }
    cudaDeviceSynchronize();
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
                            bool endpoints,
                            WT const *weights,
                            VT const number_of_sources,
                            VT const *sources) {
    //TODO(xcadet): DBG
    printf("[DBG][BC] BETWEENNESS CENTRALITY NATIVE_CUGPRAPH\n");
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

    if (number_of_sources > 0) {
      CUGRAPH_EXPECTS(sources != nullptr,
      "sources cannot be null if number_of_source is different from 0");
    }
    // Current Implementation relies on BFS
    // FIXME: For SSSP version
    // Brandes Algorithm excpets non negative weights for the accumulation
    cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
    bc.configure(result, normalize, sources, number_of_sources);
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

// TODO(xcadet) k parameter could be used to store the sice of 'vertices' data?
/**
 * @param[out]  result          array<result_t>(number_of_vertices)
 * @param[in]   normalize       bool True -> Apply normalization
 * @param[in]   endpoints (NIY) bool Include endpoints
 * @param[in]   weights   (NIY) array<WT>(number_of_edges) Weights to use
 * @param[in]   k         (NIY) Number of sources
 * @param[in]   vertices  (NIY) array<VT>(k) Sources for traversal
 */

template <typename VT, typename ET, typename WT, typename result_t>

void betweenness_centrality(experimental::GraphCSR<VT,ET,WT> const &graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT k,
                            VT const *vertices,
                            cugraph_bc_implem_t implem) {

  //
  // NOTE:  gunrock implementation doesn't yet support the unused parameters:
  //     - endpoints
  //     - weight
  //     - k
  //     - vertices
  //
  // These parameters are present in the API to support future features.
  //
  if (implem == cugraph_bc_implem_t::CUGRAPH_DEFAULT) {
    detail::betweenness_centrality(graph, result, normalize, endpoints, weight, k, vertices);
  } else if (implem == cugraph_bc_implem_t::CUGRAPH_GUNROCK) {
    gunrock::betweenness_centrality(graph, result, normalize);
  } else {
    CUGRAPH_FAIL("Invalid Betweenness Centrality implementation, please refer to cugraph_bc_implem_t for valid implementations");
  }
}

template void betweenness_centrality<int, int, float, float>(experimental::GraphCSR<int,int,float> const &, float*, bool, bool, float const *, int, int const *, cugraph_bc_implem_t);

} //namespace cugraph


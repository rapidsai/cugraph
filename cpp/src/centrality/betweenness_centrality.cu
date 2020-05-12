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

namespace cugraph {

namespace gunrock {

template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                            result_t *result,
                            bool normalize,
                            VT const *sample_seeds    = nullptr,
                            VT number_of_sample_seeds = 0)
{
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
  std::vector<ET> v_offsets(graph.number_of_vertices + 1);
  std::vector<VT> v_indices(graph.number_of_edges);
  std::vector<result_t> v_result(graph.number_of_vertices);
  std::vector<float> v_sigmas(graph.number_of_vertices);
  std::vector<int> v_labels(graph.number_of_vertices);

  // fill them
  CUDA_TRY(cudaMemcpy(v_offsets.data(),
                      graph.offsets,
                      sizeof(ET) * (graph.number_of_vertices + 1),
                      cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(
    v_indices.data(), graph.indices, sizeof(VT) * graph.number_of_edges, cudaMemcpyDeviceToHost));

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
  CUDA_TRY(cudaMemcpy(
    result, v_result.data(), sizeof(result_t) * graph.number_of_vertices, cudaMemcpyHostToDevice));

  // normalize result
  if (normalize) {
    float denominator = (graph.number_of_vertices - 1) * (graph.number_of_vertices - 2);

    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      result,
                      result + graph.number_of_vertices,
                      result,
                      [denominator] __device__(float f) { return (f * 2) / denominator; });
  } else {
    //
    //  gunrock answer needs to be doubled to match networkx
    //
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      result,
                      result + graph.number_of_vertices,
                      result,
                      [] __device__(float f) { return (f * 2); });
  }
}

}  // namespace gunrock

template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT k,
                            VT const *vertices)
{
  //
  // NOTE:  gunrock implementation doesn't yet support the unused parameters:
  //     - endpoints
  //     - weight
  //     - k
  //     - vertices
  //
  // These parameters are present in the API to support future features.
  //
  gunrock::betweenness_centrality(graph, result, normalize);
}

template void betweenness_centrality<int, int, float, float>(
  experimental::GraphCSRView<int, int, float> const &,
  float *,
  bool,
  bool,
  float const *,
  int,
  int const *);

}  // namespace cugraph

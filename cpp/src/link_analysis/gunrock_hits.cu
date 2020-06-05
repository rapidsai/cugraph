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

/**
 * ---------------------------------------------------------------------------*
 * @brief wrapper calling gunrock's HITS analytic
 * --------------------------------------------------------------------------*/

#include <algorithms.hpp>
#include <graph.hpp>

#include <utilities/error_utils.h>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>

#include <gunrock/gunrock.h>

namespace cugraph {

namespace gunrock {

template <typename vertex_t, typename edge_t, typename weight_t>
void hits(cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
          int max_iter,
          weight_t tolerance,
          weight_t const *starting_value,
          bool normalized,
          weight_t *hubs,
          weight_t *authorities)
{
  //
  //  NOTE:  gunrock doesn't support tolerance parameter
  //         gunrock doesn't support passing a starting value
  //         gunrock doesn't support the normalized parameter
  //
  //  FIXME: gunrock uses a 2-norm, while networkx uses a 1-norm.
  //         They will add a parameter to allow us to specify
  //         which norm to use.
  //
  std::vector<edge_t> local_offsets(graph.number_of_vertices + 1);
  std::vector<vertex_t> local_indices(graph.number_of_edges);
  std::vector<weight_t> local_hubs(graph.number_of_vertices);
  std::vector<weight_t> local_authorities(graph.number_of_vertices);

  //    Ideally:
  //
  //::hits(graph.number_of_vertices, graph.number_of_edges, graph.offsets, graph.indices,
  //       max_iter, hubs, authorities, DEVICE);
  //
  //    For now, the following:

  CUDA_TRY(cudaMemcpy(local_offsets.data(),
                      graph.offsets,
                      (graph.number_of_vertices + 1) * sizeof(edge_t),
                      cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(local_indices.data(),
                      graph.indices,
                      graph.number_of_edges * sizeof(vertex_t),
                      cudaMemcpyDeviceToHost));

  ::hits(graph.number_of_vertices,
         graph.number_of_edges,
         local_offsets.data(),
         local_indices.data(),
         max_iter,
         local_hubs.data(),
         local_authorities.data());

  CUDA_TRY(cudaMemcpy(
    hubs, local_hubs.data(), graph.number_of_vertices * sizeof(weight_t), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(authorities,
                      local_authorities.data(),
                      graph.number_of_vertices * sizeof(weight_t),
                      cudaMemcpyHostToDevice));
}

template void hits(cugraph::experimental::GraphCSRView<int32_t, int32_t, float> const &,
                   int,
                   float,
                   float const *,
                   bool,
                   float *,
                   float *);

}  // namespace gunrock

}  // namespace cugraph

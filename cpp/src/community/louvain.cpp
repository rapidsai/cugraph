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

#include <algorithms.hpp>
#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/sequence.h>

#include <community/louvain_kernels.hpp>

#include "utilities/error.hpp"

namespace cugraph {

template <typename VT, typename ET, typename WT>
void louvain(experimental::GraphCSRView<VT, ET, WT> const &graph,
             WT *final_modularity,
             int *num_level,
             VT *louvain_parts,
             int max_iter)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, louvain expects a weighted graph");
  CUGRAPH_EXPECTS(final_modularity != nullptr, "API error, final_modularity is null");
  CUGRAPH_EXPECTS(num_level != nullptr, "API error, num_level is null");
  CUGRAPH_EXPECTS(louvain_parts != nullptr, "API error, louvain_parts is null");

  detail::louvain<VT, ET, WT>(graph, final_modularity, num_level, louvain_parts, max_iter);
}

template void louvain(
  experimental::GraphCSRView<int32_t, int32_t, float> const &, float *, int *, int32_t *, int);
template void louvain(
  experimental::GraphCSRView<int32_t, int32_t, double> const &, double *, int *, int32_t *, int);

}  // namespace cugraph

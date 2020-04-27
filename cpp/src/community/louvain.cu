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

#include <graph.hpp>
#include <algorithms.hpp>

#include "utilities/error_utils.h"
#include <nvgraph/include/nvlouvain.cuh>

namespace cugraph {
namespace nvgraph {


template <typename VT, typename ET, typename WT>
void louvain(experimental::GraphCSR<VT, ET, WT> const &graph,
             WT *final_modularity,
             VT *num_level,
             VT *louvain_parts,
             int max_iter) {

  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, louvain expects a weighted graph");
  CUGRAPH_EXPECTS(final_modularity != nullptr, "API error, final_modularity is null");
  CUGRAPH_EXPECTS(num_level != nullptr, "API error, num_level is null");
  CUGRAPH_EXPECTS(louvain_parts != nullptr, "API error, louvain_parts is null");

  std::ostream log(0);

  bool weighted{true};

  WT mod{0.0};
  VT n_level{0};

  nvlouvain::louvain<VT,WT>(graph.offsets, graph.indices, graph.edge_data,
                            graph.number_of_vertices, graph.number_of_edges,
                            weighted, false, nullptr, mod,
                            louvain_parts, n_level, max_iter, log);

  *final_modularity = mod;
  *num_level = n_level;
}

template void louvain(experimental::GraphCSR<int32_t, int32_t, float> const &, float *, int32_t *, int32_t *, int);
template void louvain(experimental::GraphCSR<int32_t, int32_t, double> const &, double *, int32_t *, int32_t *, int);
  //template void louvain(experimental::GraphCSR<int64_t, int64_t, float> const &, float *, int64_t *, int64_t *, int);
  //template void louvain(experimental::GraphCSR<int64_t, int64_t, double> const &, double *, int64_t *, int64_t *, int);

} //namespace nvgraph
} //namespace cugraph

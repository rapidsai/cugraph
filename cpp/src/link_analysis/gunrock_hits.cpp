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

#include <utilities/error.hpp>

#include <gunrock/gunrock.h>

namespace cugraph {

namespace gunrock {

const int HOST{1};    // gunrock should expose the device constant at the API level.
const int DEVICE{2};  // gunrock should expose the device constant at the API level.

template <typename vertex_t, typename edge_t, typename weight_t>
void hits(cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
          int max_iter,
          weight_t tolerance,
          weight_t const *starting_value,
          bool normalized,
          weight_t *hubs,
          weight_t *authorities)
{
  CUGRAPH_EXPECTS(hubs != nullptr, "Invalid API parameter: hubs array should be of size V");
  CUGRAPH_EXPECTS(authorities != nullptr,
                  "Invalid API parameter: authorities array should be of size V");

  //
  //  NOTE:  gunrock doesn't support passing a starting value
  //
  ::hits(graph.number_of_vertices,
         graph.number_of_edges,
         graph.offsets,
         graph.indices,
         max_iter,
         tolerance,
         HITS_NORMALIZATION_METHOD_1,
         hubs,
         authorities,
         DEVICE);
}

template void hits(cugraph::GraphCSRView<int32_t, int32_t, float> const &,
                   int,
                   float,
                   float const *,
                   bool,
                   float *,
                   float *);

}  // namespace gunrock
}  // namespace cugraph

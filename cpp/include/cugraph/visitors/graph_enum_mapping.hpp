/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//

#pragma once

#include "graph_enum.hpp"
#include <cugraph/graph.hpp>

namespace cugraph {
namespace visitors {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool,
          bool,
          GTypes>
struct GMapType;  // primary template, purposely empty

// partial specializations:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st_tr, bool multi_gpu>
struct GMapType<vertex_t, edge_t, weight_t, st_tr, multi_gpu, GTypes::GRAPH_T> {
  using type = graph_t<vertex_t, edge_t, weight_t, st_tr, multi_gpu>;
};

}  // namespace visitors
}  // namespace cugraph

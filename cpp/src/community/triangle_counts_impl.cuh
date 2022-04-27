/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#pragma once

#include <cugraph/algorithms.hpp>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void triangle_counts(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                     std::optional<raft::device_span<vertex_t>> vertices,
                     raft::device_span<edge_t> counts,
                     bool do_expensive_check)
{
  CUGRAPH_FAIL("unimplemented.");

  // 1. Check input arguments.

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input arguments: triangle_counts currently supports undirected graphs only.");
  if (vertices) {
    CUGRAPH_EXPECTS(counts.size() == (*vertices).size(),
                    "Invalid arguments: counts.size() does not coincide with (*vertices).size().");
  } else {
    CUGRAPH_EXPECTS(counts.size() == (*vertices).size(),
                    "Invalid arguments: counts.size() does not coincide with (*vertices).size().");
  }

  if (do_expensive_check) {
    if (vertices) {}
  }

  // 2. Mask-out self-loops (FIXME: this can be implemented much more efficiently once we support
  // edge masking).

  // 3. Find 2-core.

  // 4. Keep only low-degree to high-degree edges.

  // 5. Neighbor intersection in chunks

  // 6. If vertices.has_value(), gather triangle counts (FIXME: This is inefficient if
  // (*vertices).size() is small, we may better extract a subgraph including only the edges between
  // *vertices and (*vertices)'s one-hop neighbors and count triangles using the subgraph).

  return;
}

}  // namespace cugraph

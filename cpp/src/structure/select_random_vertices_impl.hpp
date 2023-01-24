/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

namespace cugraph {

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<vertex_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state,
  vertex_t select_count,
  bool with_replacement)
{
#ifndef NO_CUGRAPH_OPS
  CUGRAPH_EXPECTS(with_replacement || (select_count <= graph_view.number_of_vertices()),
                  "Requested more vertices than can be selected");

  rmm::device_uvector<vertex_t> result(select_count, handle.get_stream());

  if (with_replacement) {
    detail::uniform_random_fill(handle.get_stream(),
                                result.data(),
                                result.size(),
                                vertex_t{0},
                                graph_view.number_of_vertices(),
                                rng_state);
  } else {
    if (select_count == graph_view.number_of_vertices()) {
      detail::sequence_fill(handle.get_stream(), result.data(), result.size(), vertex_t{0});
    } else {
      rmm::device_scalar<vertex_t> d_num_vertices(graph_view.number_of_vertices(),
                                                  handle.get_stream());

      cugraph::ops::gnn::graph::get_sampling_index(result.data(),
                                                   rng_state,
                                                   d_num_vertices.data(),
                                                   1,
                                                   select_count,
                                                   with_replacement,
                                                   handle.get_stream());
    }
  }

  return result;
#else
  CUGRAPH_FAIL("select_random_vertices not supported if CUGRAPH_OPS is not available");
#endif
}

}  // namespace cugraph

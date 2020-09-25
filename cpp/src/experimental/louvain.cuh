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
#pragma once

#include <experimental/graph.hpp>

namespace cugraph {
namespace experimental {

template <typename graph_view_type>
class Louvain {
 public:
  using graph_view_t = graph_view_type;
  using vertex_t     = typename graph_view_t::vertex_type;
  using edge_t       = typename graph_view_t::edge_type;
  using weight_t     = typename graph_view_t::weight_type;
  using graph_t      = experimental::graph_t<vertex_t,
                                        edge_t,
                                        weight_t,
                                        graph_view_t::is_adj_matrix_transposed,
                                        graph_view_t::is_multi_gpu>;

  Louvain(raft::handle_t const &handle, graph_view_t const &graph_view)
    : handle_(handle), current_graph_view_(graph_view)
  {
  }

  virtual std::pair<size_t, weight_t> operator()(vertex_t *d_cluster_vec,
                                                 size_t max_level,
                                                 weight_t resolution)
  {
    CUGRAPH_FAIL("unimplemented");
  }

 protected:
  raft::handle_t const &handle_;
  graph_view_t current_graph_view_;
};

}  // namespace experimental
}  // namespace cugraph

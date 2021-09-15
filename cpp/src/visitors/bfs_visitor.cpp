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

#include <cugraph/algorithms.hpp>
#include <cugraph/visitors/bfs_visitor.hpp>

namespace cugraph {
namespace visitors {

//
// wrapper code:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
void bfs_visitor<vertex_t,
                 edge_t,
                 weight_t,
                 st,  // FIXME: can only be false for BFS
                 mg,
                 std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  visit_graph(graph_envelope_t::base_graph_t const& graph)
{
  // Note: this must be called only on:
  // graph_view_t<vertex_t, edge_t, weight_t, false, mg>
  //
  if constexpr (st == false) {
    // unless algorithms only call virtual graph methods
    // under the hood, the algos require this conversion:
    //
    graph_t<vertex_t, edge_t, weight_t, false, mg> const* p_g =
      static_cast<graph_t<vertex_t, edge_t, weight_t, st, mg> const*>(&graph);

    auto gview = p_g->view();

    auto const& v_args = ep_.get_args();

    // unpack bfs() args:
    //
    assert(v_args.size() == 8);

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    vertex_t* p_d_dist = static_cast<vertex_t*>(v_args[1]);

    vertex_t* p_d_predec = static_cast<vertex_t*>(v_args[2]);

    vertex_t const* p_d_src = static_cast<vertex_t const*>(v_args[3]);

    size_t n_sources = *static_cast<size_t*>(v_args[4]);

    bool dir_opt = *static_cast<bool*>(v_args[5]);

    auto depth_l = *static_cast<vertex_t*>(v_args[6]);

    bool check = *static_cast<bool*>(v_args[7]);

    // call algorithm
    // (no result; void)
    //
    bfs(handle, gview, p_d_dist, p_d_predec, p_d_src, n_sources, dir_opt, depth_l, check);
  } else {
    CUGRAPH_FAIL("Unsupported BFS algorithm (store_transposed == true).");
  }
}

// EIDir's:
//
template class bfs_visitor<int, int, float, true, true>;
template class bfs_visitor<int, int, double, true, true>;

template class bfs_visitor<int, int, float, true, false>;
template class bfs_visitor<int, int, double, true, false>;

template class bfs_visitor<int, int, float, false, true>;
template class bfs_visitor<int, int, double, false, true>;

template class bfs_visitor<int, int, float, false, false>;
template class bfs_visitor<int, int, double, false, false>;

//------

template class bfs_visitor<int, long, float, true, true>;
template class bfs_visitor<int, long, double, true, true>;

template class bfs_visitor<int, long, float, true, false>;
template class bfs_visitor<int, long, double, true, false>;

template class bfs_visitor<int, long, float, false, true>;
template class bfs_visitor<int, long, double, false, true>;

template class bfs_visitor<int, long, float, false, false>;
template class bfs_visitor<int, long, double, false, false>;

//------

template class bfs_visitor<long, long, float, true, true>;
template class bfs_visitor<long, long, double, true, true>;

template class bfs_visitor<long, long, float, true, false>;
template class bfs_visitor<long, long, double, true, false>;

template class bfs_visitor<long, long, float, false, true>;
template class bfs_visitor<long, long, double, false, true>;

template class bfs_visitor<long, long, float, false, false>;
template class bfs_visitor<long, long, double, false, false>;

}  // namespace visitors

namespace api {
using namespace cugraph::visitors;
// wrapper:
// macro option: MAKE_WRAPPER(bfs)
//
return_t bfs(graph_envelope_t const& g, erased_pack_t& ep)
{
  auto p_visitor = g.factory()->make_bfs_visitor(ep);

  g.apply(*p_visitor);

  return p_visitor->get_result();
}

}  // namespace api
}  // namespace cugraph

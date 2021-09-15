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

#include <cugraph/algorithms.hpp>
#include <cugraph/visitors/rw_visitor.hpp>

namespace cugraph {
namespace visitors {
//
// wrapper code:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
void rw_visitor<vertex_t,
                edge_t,
                weight_t,
                st,
                mg,
                std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  visit_graph(graph_envelope_t::base_graph_t const& graph)
{
  using index_t      = edge_t;
  using ptr_params_t = std::unique_ptr<sampling_params_t>;

  // Note: this must be called only on:
  // graph_view_t<vertex_t, edge_t, weight_t, false, mg>
  //
  if constexpr (st == false && mg == false) {
    // unless algorithms only call virtual graph methods
    // under the hood, the algos require this conversion:
    //
    graph_t<vertex_t, edge_t, weight_t, false, mg> const* p_g =
      static_cast<graph_t<vertex_t, edge_t, weight_t, st, mg> const*>(&graph);

    auto gview = p_g->view();

    auto const& v_args = ep_.get_args();

    // unpack bfs() args:
    //
    assert(v_args.size() == 6);

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    vertex_t* p_d_start = static_cast<vertex_t*>(v_args[1]);

    index_t num_paths = *static_cast<index_t*>(v_args[2]);

    index_t max_depth = *static_cast<index_t*>(v_args[3]);

    bool use_padding = *static_cast<bool*>(v_args[4]);

    ptr_params_t p_uniq_params = std::move(*static_cast<ptr_params_t*>(v_args[5]));

    // call algorithm
    //
    auto tpl_result = random_walks(
      handle, gview, p_d_start, num_paths, max_depth, use_padding, std::move(p_uniq_params));

    auto tpl_erased_result = std::make_tuple(std::get<0>(tpl_result).release(),
                                             std::get<1>(tpl_result).release(),
                                             std::get<2>(tpl_result).release());

    result_ = return_t{std::move(tpl_erased_result)};
  } else {
    CUGRAPH_FAIL(
      "Unsupported RandomWalks algorithm (store_transposed == true or multi_gpu == true).");
  }
}

// EIDir's:
//
template class rw_visitor<int, int, float, true, true>;
template class rw_visitor<int, int, double, true, true>;

template class rw_visitor<int, int, float, true, false>;
template class rw_visitor<int, int, double, true, false>;

template class rw_visitor<int, int, float, false, true>;
template class rw_visitor<int, int, double, false, true>;

template class rw_visitor<int, int, float, false, false>;
template class rw_visitor<int, int, double, false, false>;

//------

template class rw_visitor<int, long, float, true, true>;
template class rw_visitor<int, long, double, true, true>;

template class rw_visitor<int, long, float, true, false>;
template class rw_visitor<int, long, double, true, false>;

template class rw_visitor<int, long, float, false, true>;
template class rw_visitor<int, long, double, false, true>;

template class rw_visitor<int, long, float, false, false>;
template class rw_visitor<int, long, double, false, false>;

//------

template class rw_visitor<long, long, float, true, true>;
template class rw_visitor<long, long, double, true, true>;

template class rw_visitor<long, long, float, true, false>;
template class rw_visitor<long, long, double, true, false>;

template class rw_visitor<long, long, float, false, true>;
template class rw_visitor<long, long, double, false, true>;

template class rw_visitor<long, long, float, false, false>;
template class rw_visitor<long, long, double, false, false>;

}  // namespace visitors

namespace api {
using namespace cugraph::visitors;
// wrapper:
// macro option: MAKE_WRAPPER(bfs)
//
return_t random_walks(graph_envelope_t const& g, erased_pack_t& ep)
{
  auto p_visitor = g.factory()->make_rw_visitor(ep);

  g.apply(*p_visitor);

  return p_visitor->get_result();
}

}  // namespace api

}  // namespace cugraph

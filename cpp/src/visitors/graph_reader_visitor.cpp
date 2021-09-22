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

#include <cugraph/visitors/graph_reader_visitor.hpp>

namespace cugraph {
namespace visitors {
//
// wrapper code:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
void graph_reader_visitor_t<vertex_t,
                            edge_t,
                            weight_t,
                            st,
                            mg,
                            std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  visit_graph(graph_envelope_t::base_graph_t const& graph)
{
  using index_t = edge_t;

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
    // assert(v_args.size() == ?); // raft::handle_t, char* path, bool is_weighted...,

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    char const* p_path = static_cast<char*>(v_args[1]);

    bool is_weighted = *static_cast<bool*>(v_args[2]);

    // TODO:
    //

    // read graph:
    //

    // create graph_envelope_t

    /// result_ = return_t{std::move(graph_envelope)};
  } else {
    CUGRAPH_FAIL("Unsupported graph reader (store_transposed == true or multi_gpu == true).");
  }
}

// EIDir's:
//
template class graph_reader_visitor_t<int, int, float, true, true>;
template class graph_reader_visitor_t<int, int, double, true, true>;

template class graph_reader_visitor_t<int, int, float, true, false>;
template class graph_reader_visitor_t<int, int, double, true, false>;

template class graph_reader_visitor_t<int, int, float, false, true>;
template class graph_reader_visitor_t<int, int, double, false, true>;

template class graph_reader_visitor_t<int, int, float, false, false>;
template class graph_reader_visitor_t<int, int, double, false, false>;

//------

template class graph_reader_visitor_t<int, long, float, true, true>;
template class graph_reader_visitor_t<int, long, double, true, true>;

template class graph_reader_visitor_t<int, long, float, true, false>;
template class graph_reader_visitor_t<int, long, double, true, false>;

template class graph_reader_visitor_t<int, long, float, false, true>;
template class graph_reader_visitor_t<int, long, double, false, true>;

template class graph_reader_visitor_t<int, long, float, false, false>;
template class graph_reader_visitor_t<int, long, double, false, false>;

//------

template class graph_reader_visitor_t<long, long, float, true, true>;
template class graph_reader_visitor_t<long, long, double, true, true>;

template class graph_reader_visitor_t<long, long, float, true, false>;
template class graph_reader_visitor_t<long, long, double, true, false>;

template class graph_reader_visitor_t<long, long, float, false, true>;
template class graph_reader_visitor_t<long, long, double, false, true>;

template class graph_reader_visitor_t<long, long, float, false, false>;
template class graph_reader_visitor_t<long, long, double, false, false>;

}  // namespace visitors

namespace api {
using namespace cugraph::visitors;
// wrapper:
//
return_t graph_read_mm_file(
  DTypes vertex_tid, DTypes edge_tid, DTypes weight_tid, bool st, bool mg, erased_pack_t& ep_reader)
{
  auto const& v_args = ep_reader.get_args();

  // unpack args:
  //
  assert(v_args.size() > 0);  // raft::handle_t, ...

  // cnstr. args unpacking:
  //
  raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

  erased_pack_t ep_graph{const_cast<raft::handle_t*>(&handle)};
  GTypes graph_tid = GTypes::GRAPH_T;

  graph_envelope_t g{vertex_tid, edge_tid, weight_tid, st, mg, graph_tid, ep_graph};

  auto p_visitor = g.factory()->make_graph_reader_visitor(ep_reader);

  g.apply(*p_visitor);

  return p_visitor->get_result();
}

}  // namespace api

}  // namespace cugraph

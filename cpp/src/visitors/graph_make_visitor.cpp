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
#include <cugraph/visitors/graph_make_visitor.hpp>

namespace cugraph {
namespace visitors {
//
// wrapper code:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
void graph_maker_visitor<vertex_t,
                         edge_t,
                         weight_t,
                         st,
                         mg,
                         std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  visit_graph(graph_envelope_t::base_graph_t const& graph)
{
  if constexpr (mg == false) {
    auto const& v_args = ep_.get_args();

    auto num_args = v_args.size();
    assert(num_args > 6);

    // cnstr. args unpacking:
    //
    raft::handle_t* ptr_handle = static_cast<raft::handle_t*>(v_args[0]);
    vertex_t const* p_src      = static_cast<vertex_t*>(v_args[1]);
    vertex_t const* p_dst      = static_cast<vertex_t*>(v_args[2]);
    weight_t const* p_weights  = static_cast<weight_t*>(v_args[3]);
    edge_t num_edges           = *static_cast<edge_t*>(v_args[4]);
    vertex_t num_vertices      = *static_cast<vertex_t*>(v_args[5]);
    bool check                 = *static_cast<bool*>(v_args[6]);

    bool is_sym{false};
    bool is_multigraph{false};

    if (num_args > 7) {
      is_sym = *static_cast<bool*>(v_args[7]);
      if (num_args > 8) is_multigraph = *static_cast<bool*>(v_args[8]);
    }

    cugraph::graph_properties_t graph_props{is_sym, is_multigraph};

    std::optional<weight_t const*> opt_ptr_w =
      p_weights != nullptr ? std::optional{p_weights} : std::nullopt;

    cugraph::edgelist_t<vertex_t, edge_t, weight_t> edgelist{p_src, p_dst, opt_ptr_w, num_edges};

    cugraph::graph_meta_t<vertex_t, edge_t, false> meta{num_vertices, graph_props, std::nullopt};
    erased_pack_t ep_graph{ptr_handle, &edgelist, &meta, &check};

    DTypes vertex_tid = reverse_dmap_t<vertex_t>::type_id;
    DTypes edge_tid   = reverse_dmap_t<edge_t>::type_id;
    DTypes weight_tid = reverse_dmap_t<weight_t>::type_id;
    bool st_id        = st;
    bool mg_id        = mg;
    GTypes graph_tid  = GTypes::GRAPH_T;

    graph_envelope_t graph_envelope{
      vertex_tid, edge_tid, weight_tid, st_id, mg_id, graph_tid, ep_graph};

    result_ = return_t{std::move(graph_envelope)};

  } else {
    CUGRAPH_FAIL("Graph factory visitor not currently supported (multi_gpu == true).");
  }
}

// EIDir's:
//
template class graph_maker_visitor<int, int, float, true, true>;
template class graph_maker_visitor<int, int, double, true, true>;

template class graph_maker_visitor<int, int, float, true, false>;
template class graph_maker_visitor<int, int, double, true, false>;

template class graph_maker_visitor<int, int, float, false, true>;
template class graph_maker_visitor<int, int, double, false, true>;

template class graph_maker_visitor<int, int, float, false, false>;
template class graph_maker_visitor<int, int, double, false, false>;

//------

template class graph_maker_visitor<int, long, float, true, true>;
template class graph_maker_visitor<int, long, double, true, true>;

template class graph_maker_visitor<int, long, float, true, false>;
template class graph_maker_visitor<int, long, double, true, false>;

template class graph_maker_visitor<int, long, float, false, true>;
template class graph_maker_visitor<int, long, double, false, true>;

template class graph_maker_visitor<int, long, float, false, false>;
template class graph_maker_visitor<int, long, double, false, false>;

//------

template class graph_maker_visitor<long, long, float, true, true>;
template class graph_maker_visitor<long, long, double, true, true>;

template class graph_maker_visitor<long, long, float, true, false>;
template class graph_maker_visitor<long, long, double, true, false>;

template class graph_maker_visitor<long, long, float, false, true>;
template class graph_maker_visitor<long, long, double, false, true>;

template class graph_maker_visitor<long, long, float, false, false>;
template class graph_maker_visitor<long, long, double, false, false>;

}  // namespace visitors

namespace api {
using namespace cugraph::visitors;
// wrapper:
//
return_t graph_create(
  DTypes vertex_tid, DTypes edge_tid, DTypes weight_tid, bool st, bool mg, erased_pack_t& ep_cnstr)
{
  try {
    auto const& v_args = ep_cnstr.get_args();

    // unpack args:
    //
    assert(v_args.size() > 0);  // raft::handle_t, ...

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    erased_pack_t ep_graph{const_cast<raft::handle_t*>(&handle)};
    GTypes graph_tid = GTypes::GRAPH_T;

    // first construct empty graph,
    // to be able to resolve types at runtime (CDD):
    //
    graph_envelope_t g{vertex_tid, edge_tid, weight_tid, false, false, graph_tid, ep_graph};

    auto p_visitor = g.factory()->make_graph_maker_visitor(ep_cnstr);

    g.apply(*p_visitor);

    // graph_envelope_t can be extracted through the following mechanism:
    //
    // return_t::base_return_t* p_base_ret = p_visitor->get_result().release();
    // return_t::generic_return_t<graph_envelope_t>* p_typed_ret =
    //        dynamic_cast<return_t::generic_return_t<graph_envelope_t>*>(p_base_ret);
    // graph_envelope_t const& graph_envelope = p_typed_ret->get();
    //
    return p_visitor->get_result();

  } catch (std::exception const& ex) {
    std::cerr << "cugraph++: " << ex.what() << "in graph_envelope factory.\n";
    return return_t{};
  } catch (...) {
    std::cerr << "cugraph++: Unknown exception occurred in graph_envelope factory.\n";
    return return_t{};
  }
}

}  // namespace api

}  // namespace cugraph

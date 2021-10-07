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

#include <stdexcept>
#include <tuple>

#include "graph_envelope.hpp"
// prevent clang-format to rearange order of headers
#include "erased_pack.hpp"
//
// not really needed here;
// just to make happy the clang-format policy
// of header inclusion to be order-independent...
//
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>

#define _DEBUG_

#ifdef _DEBUG_
#include <iostream>
#endif

namespace cugraph {
namespace visitors {

struct graph_factory_base_t {
  virtual ~graph_factory_base_t(void) {}

  virtual std::unique_ptr<graph_envelope_t::base_graph_t> make_graph(erased_pack_t&) const = 0;
};

// argument unpacker (from `erased_pack_t`)
// for graph construction
//
template <typename graph_type>
struct graph_arg_unpacker_t {
  using vertex_t           = typename graph_type::vertex_type;
  using edge_t             = typename graph_type::edge_type;
  using weight_t           = typename graph_type::weight_type;
  static constexpr bool mg = graph_type::is_multi_gpu;

  void operator()(erased_pack_t& ep,
                  std::tuple<raft::handle_t const&,
                             vertex_t*,
                             vertex_t*,
                             weight_t*,
                             vertex_t*,
                             edge_t,
                             vertex_t,
                             edge_t,
                             bool>& t_args) const
  {
  }
};

// primary template factory; to be (partiallY) specialized;
// and explicitly instantiated for concrete graphs
//
template <typename graph_type>
struct graph_factory_t : graph_factory_base_t {
  std::unique_ptr<graph_envelope_t::base_graph_t> make_graph(erased_pack_t&) const override
  {
    throw std::runtime_error("Empty factory, not to be called...");
  }
};

// Linker PROBLEM (FIXED):
// dispatcher needs _ALL_ paths instantiated,
// not just the ones explicitly instantiated
// (EIDir) in `graph.cpp`
//
// Posiible SOLUTIONS:
//
// (1.) the _factory_ must provide "dummy"
//      instantiations for paths not needed;
//
// or:
//
// (2.) (Adopted solution)
//      the _dispatcher_ (graph_dispatcher())
//      must provide empty implementation
//      for the instantiations that are not needed; (Done!)
//
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
struct graph_factory_t<graph_t<vertex_t, edge_t, weight_t, multi_gpu, std::enable_if_t<multi_gpu>>>
  : graph_factory_base_t {
  std::unique_ptr<graph_envelope_t::base_graph_t> make_graph(erased_pack_t& ep) const override
  {
    /// std::cout << "Multi-GPU factory.\n";
    std::vector<void*> const& v_args{ep.get_args()};

    // invoke cnstr. using cython arg pack:
    //
    assert(v_args.size() == 4);

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    auto const& edgelists =
      *static_cast<std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const*>(v_args[1]);

    auto meta = *static_cast<graph_meta_t<vertex_t, edge_t, multi_gpu> const*>(v_args[2]);

    bool check = *static_cast<bool*>(v_args[3]);

    return std::make_unique<graph_t<vertex_t, edge_t, weight_t, multi_gpu>>(
      handle, edgelists, meta, check);
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
struct graph_factory_t<graph_t<vertex_t, edge_t, weight_t, multi_gpu, std::enable_if_t<!multi_gpu>>>
  : graph_factory_base_t {
  std::unique_ptr<graph_envelope_t::base_graph_t> make_graph(erased_pack_t& ep) const override
  {
    /// std::cout << "Single-GPU factory.\n";
    std::vector<void*> const& v_args{ep.get_args()};

    assert(v_args.size() == 4);

    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    auto const& elist = *static_cast<edgelist_t<vertex_t, edge_t, weight_t> const*>(v_args[1]);

    auto meta = *static_cast<graph_meta_t<vertex_t, edge_t, multi_gpu> const*>(v_args[2]);

    bool check = *static_cast<bool*>(v_args[3]);

    return std::make_unique<graph_t<vertex_t, edge_t, weight_t, multi_gpu>>(
      handle, elist, meta, check);
  }
};

}  // namespace visitors
}  // namespace cugraph

/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
  static constexpr bool st = graph_type::is_storage_transposed;
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
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
struct graph_factory_t<
  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>>
  : graph_factory_base_t {
  std::unique_ptr<graph_envelope_t::base_graph_t> make_graph(erased_pack_t& ep) const override
  {
#ifdef _DEBUG_
    std::cout << "Enter graph factory...\n";
#endif

    /// std::cout << "Multi-GPU factory.\n";
    std::vector<void*> const& v_args{ep.get_args()};

    // branch on various constructors based on
    // number of arguments in the pack:
    //
    auto pack_arg_sz = v_args.size();
    assert(pack_arg_sz > 0);  // need at least the raft handle

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    if (pack_arg_sz == 1) {
      // invoke graph_t(handle);
      return std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
        handle);
    } else {
      // invoke cnstr. using cython arg pack:
      //
      assert(pack_arg_sz == 9);

      vertex_t* src_vertices             = static_cast<vertex_t*>(v_args[1]);
      vertex_t* dst_vertices             = static_cast<vertex_t*>(v_args[2]);
      weight_t* weights                  = static_cast<weight_t*>(v_args[3]);
      vertex_t* vertex_partition_offsets = static_cast<vertex_t*>(v_args[4]);
      edge_t num_partition_edges         = *static_cast<edge_t*>(v_args[5]);
      vertex_t num_global_vertices       = *static_cast<vertex_t*>(v_args[6]);
      edge_t num_global_edges            = *static_cast<edge_t*>(v_args[7]);
      bool sorted_by_degree              = *static_cast<bool*>(v_args[8]);

      // TODO: un-hardcode: have it passed int `ep`
      //
      graph_properties_t graph_props{.is_symmetric = false, .is_multigraph = false};
      bool do_expensive_check{false};  // FIXME: check what should this default to

      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();  // pcols
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();  // prows

      std::vector<edgelist_t<vertex_t, edge_t, weight_t>> edgelist(
        {{src_vertices, dst_vertices, weights, num_partition_edges}});

      std::vector<vertex_t> partition_offsets_vector(
        vertex_partition_offsets, vertex_partition_offsets + (row_comm_size * col_comm_size) + 1);

      partition_t<vertex_t> partition(
        partition_offsets_vector, row_comm_size, col_comm_size, row_comm_rank, col_comm_rank);

      std::optional<std::vector<vertex_t>>
        opt_seg_off{};  // FIXME: may needd to pass/extract segment_offsets vector

      graph_meta_t<vertex_t, edge_t, multi_gpu> meta{
        num_global_vertices, num_global_edges, graph_props, partition, opt_seg_off};

      return std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
        handle, edgelist, meta, do_expensive_check);
    }
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
struct graph_factory_t<
  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>>
  : graph_factory_base_t {
  std::unique_ptr<graph_envelope_t::base_graph_t> make_graph(erased_pack_t& ep) const override
  {
    /// std::cout << "Single-GPU factory.\n";

    std::vector<void*> const& v_args{ep.get_args()};

    // branch on various constructors based on
    // number of arguments in the pack:
    //
    auto pack_arg_sz = v_args.size();
    assert(pack_arg_sz > 0);  // need at least the raft handle

    // cnstr. args unpacking:
    //
    raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

    if (pack_arg_sz == 1) {
      // invoke graph_t(handle);
      return std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
        handle);
    } else {
      assert(pack_arg_sz == 4);

      raft::handle_t const& handle = *static_cast<raft::handle_t const*>(v_args[0]);

      auto const& elist = *static_cast<edgelist_t<vertex_t, edge_t, weight_t> const*>(v_args[1]);

      auto meta = *static_cast<graph_meta_t<vertex_t, edge_t, multi_gpu> const*>(v_args[2]);

      bool check = *static_cast<bool*>(v_args[3]);

      return std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
        handle, elist, meta, check);
    }
  }
};

}  // namespace visitors
}  // namespace cugraph

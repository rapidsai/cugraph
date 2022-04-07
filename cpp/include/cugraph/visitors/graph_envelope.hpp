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

/**
 * @brief Set of classes abstracting the type-erasure, templates, and template constraints
 *        to client code that must supply run-time type information (RTTI) and has no template
constructs.
 *
 *  Goal: be able to call an algorithm (say. louvain() on a type erased graph created from RTTI:
 * {
 *  auto graph = make_graph(flags...);
 *  auto res = louvain(graph, params...);
 * }
 * params will be also type-erased (or same type regardless of graph-type); and will
 * be appropriately passed to the Factory and then converted and passed to Visitor constructor
*/

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "enum_mapping.hpp"
#include "graph_enum.hpp"

#include <cugraph/utilities/graph_traits.hpp>

namespace cugraph {
namespace visitors {

class erased_pack_t;  // forward...
class return_t;       // forward...

// visitor base, incomplete:
//
class visitor_t;  // forward...

// envelope class around all
// graph classes:
//
struct graph_envelope_t {
  struct base_graph_t {  // necessary to avoid circular dependency
                         // between graph_base_t and graph_envelope_t
    virtual ~base_graph_t() {}

    /// virtual void print(void) const = 0;

    virtual void apply(visitor_t& v) const = 0;
  };

  // abstract factory:
  //
  struct visitor_factory_t {
    virtual std::unique_ptr<visitor_t> make_louvain_visitor(erased_pack_t&) const = 0;

    virtual std::unique_ptr<visitor_t> make_bfs_visitor(erased_pack_t&) const = 0;

    virtual std::unique_ptr<visitor_t> make_rw_visitor(erased_pack_t&) const = 0;

    virtual std::unique_ptr<visitor_t> make_graph_maker_visitor(erased_pack_t&) const = 0;
  };

  using pair_uniques_t =
    std::pair<std::unique_ptr<base_graph_t>, std::unique_ptr<visitor_factory_t>>;

  void apply(visitor_t& v) const
  {
    if (p_impl_fact_.first)
      p_impl_fact_.first->apply(v);
    else
      throw std::runtime_error("ERROR: Implementation not allocated.");
  }

  // void print(void) const
  // {
  //   if (p_impl_fact_.first)
  //     p_impl_fact_.first->print();
  //   else
  //     throw std::runtime_error("ERROR: Implementation not allocated.");
  // }

  std::unique_ptr<base_graph_t> const& graph(void) const { return p_impl_fact_.first; }

  std::unique_ptr<visitor_factory_t> const& factory(void) const { return p_impl_fact_.second; }

  graph_envelope_t(DTypes vertex_tid,
                   DTypes edge_tid,
                   DTypes weight_tid,
                   bool,
                   bool,
                   GTypes graph_tid,
                   erased_pack_t&);

 private:
  // need it to hide the parameterization of
  // (graph implementation, factory implementation)
  // by dependent types: vertex_t, edge_t, weight_t
  //
  pair_uniques_t p_impl_fact_;
};

// visitor base:
//
class visitor_t {
 public:
  virtual ~visitor_t(void) {}

  virtual void visit_graph(graph_envelope_t::base_graph_t const&) = 0;

  virtual return_t const& get_result(void) const = 0;

  virtual return_t&& get_result(void) = 0;
};

// convenience templatized base:
//
template <typename vertex_t, typename edge_t, typename weight_t>
struct dependent_graph_t : graph_envelope_t::base_graph_t {
  using vertex_type = vertex_t;
  using edge_type   = edge_t;
  using weight_type = weight_t;
};

// primary empty template:
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool st,
          bool mg,
          typename Enable = void>
struct dependent_factory_t;

// dummy out non-candidate instantiation paths:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
struct dependent_factory_t<vertex_t,
                           edge_t,
                           weight_t,
                           st,
                           mg,
                           std::enable_if_t<!is_candidate<vertex_t, edge_t, weight_t>::value>>
  : graph_envelope_t::visitor_factory_t {
  using vertex_type = vertex_t;
  using edge_type   = edge_t;
  using weight_type = weight_t;

  std::unique_ptr<visitor_t> make_louvain_visitor(erased_pack_t&) const override { return nullptr; }

  std::unique_ptr<visitor_t> make_bfs_visitor(erased_pack_t&) const override { return nullptr; }

  std::unique_ptr<visitor_t> make_rw_visitor(erased_pack_t&) const override { return nullptr; }

  std::unique_ptr<visitor_t> make_graph_maker_visitor(erased_pack_t&) const override
  {
    return nullptr;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
struct dependent_factory_t<vertex_t,
                           edge_t,
                           weight_t,
                           st,
                           mg,
                           std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>
  : graph_envelope_t::visitor_factory_t {
  using vertex_type = vertex_t;
  using edge_type   = edge_t;
  using weight_type = weight_t;

  std::unique_ptr<visitor_t> make_louvain_visitor(erased_pack_t&) const override;

  std::unique_ptr<visitor_t> make_bfs_visitor(erased_pack_t&) const override;

  std::unique_ptr<visitor_t> make_rw_visitor(erased_pack_t&) const override;

  std::unique_ptr<visitor_t> make_graph_maker_visitor(erased_pack_t&) const override;
};

// utility factory selector:
//
template <typename graph_type>
std::unique_ptr<visitor_t> make_visitor(
  graph_type const& tag,  // necessary to extract dependent types
  std::function<std::unique_ptr<visitor_t>(graph_envelope_t::visitor_factory_t const&,
                                           erased_pack_t&)>
    f,  // selector functor that picks up the make memf of the visitor_factory and passes `ep` to it
  erased_pack_t& ep)  // erased pack of args to be passed to factory
{
  using vertex_t    = typename graph_type::vertex_type;
  using edge_t      = typename graph_type::edge_type;
  using weight_t    = typename graph_type::weight_type;
  constexpr bool st = graph_type::is_storage_transposed;
  constexpr bool mg = graph_type::is_multi_gpu;

  dependent_factory_t<vertex_t, edge_t, weight_t, st, mg> factory;

  return f(factory, ep);
}

}  // namespace visitors
}  // namespace cugraph

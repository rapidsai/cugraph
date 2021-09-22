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

#pragma once
#include "erased_pack.hpp"
#include "graph_envelope.hpp"
#include "ret_terased.hpp"

namespace cugraph {
namespace visitors {
// primary empty template:
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool st,
          bool mg,
          typename Enable = void>
struct graph_maker_visitor;

// dummy out non-candidate instantiation paths:
//
template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
struct graph_maker_visitor<vertex_t,
                           edge_t,
                           weight_t,
                           st,
                           mg,
                           std::enable_if_t<(!is_candidate<vertex_t, edge_t, weight_t>::value)>>
  : visitor_t {
  void visit_graph(graph_envelope_t::base_graph_t const&) override
  {
    // purposely empty
  }
  return_t const& get_result(void) const override
  {
    static return_t r{};
    return r;
  }

  return_t&& get_result(void) override
  {
    static return_t r{};
    return std::move(r);
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
struct graph_maker_visitor<vertex_t,
                           edge_t,
                           weight_t,
                           st,
                           mg,
                           std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>
  : visitor_t {
  graph_maker_visitor(erased_pack_t& ep) : ep_(ep) {}

  void visit_graph(graph_envelope_t::base_graph_t const&) override;

  return_t const& get_result(void) const override { return result_; }
  return_t&& get_result(void) override { return std::move(result_); }

 private:
  erased_pack_t& ep_;
  return_t result_;
};

}  // namespace visitors
}  // namespace cugraph

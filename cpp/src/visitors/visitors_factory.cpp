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

#include <cugraph/visitors/bfs_visitor.hpp>
#include <cugraph/visitors/graph_make_visitor.hpp>
#include <cugraph/visitors/rw_visitor.hpp>

#include <cugraph/visitors/graph_envelope.hpp>

namespace cugraph {
namespace visitors {

template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
std::unique_ptr<visitor_t>
dependent_factory_t<vertex_t,
                    edge_t,
                    weight_t,
                    st,
                    mg,
                    std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  make_louvain_visitor(erased_pack_t& ep) const
{
  /// return std::unique_ptr<visitor_t>(
  ///  static_cast<visitor_t*>(new louvain_visitor<vertex_t, edge_t, weight_t, st, mg>(ep)));

  return nullptr;  // for now...
}

template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
std::unique_ptr<visitor_t>
dependent_factory_t<vertex_t,
                    edge_t,
                    weight_t,
                    st,
                    mg,
                    std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  make_bfs_visitor(erased_pack_t& ep) const
{
  return std::make_unique<bfs_visitor<vertex_t, edge_t, weight_t, st, mg>>(ep);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
std::unique_ptr<visitor_t> dependent_factory_t<
  vertex_t,
  edge_t,
  weight_t,
  st,
  mg,
  std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::make_rw_visitor(erased_pack_t&
                                                                                        ep) const
{
  return std::make_unique<rw_visitor<vertex_t, edge_t, weight_t, st, mg>>(ep);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool st, bool mg>
std::unique_ptr<visitor_t>
dependent_factory_t<vertex_t,
                    edge_t,
                    weight_t,
                    st,
                    mg,
                    std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value>>::
  make_graph_maker_visitor(erased_pack_t& ep) const
{
  return std::make_unique<graph_maker_visitor<vertex_t, edge_t, weight_t, st, mg>>(ep);
}

// EIDir's:
//
template class dependent_factory_t<int, int, float, true, true>;
template class dependent_factory_t<int, int, double, true, true>;

template class dependent_factory_t<int, int, float, true, false>;
template class dependent_factory_t<int, int, double, true, false>;

template class dependent_factory_t<int, int, float, false, true>;
template class dependent_factory_t<int, int, double, false, true>;

template class dependent_factory_t<int, int, float, false, false>;
template class dependent_factory_t<int, int, double, false, false>;

//------

template class dependent_factory_t<int, long, float, true, true>;
template class dependent_factory_t<int, long, double, true, true>;

template class dependent_factory_t<int, long, float, true, false>;
template class dependent_factory_t<int, long, double, true, false>;

template class dependent_factory_t<int, long, float, false, true>;
template class dependent_factory_t<int, long, double, false, true>;

template class dependent_factory_t<int, long, float, false, false>;
template class dependent_factory_t<int, long, double, false, false>;

//------

template class dependent_factory_t<long, long, float, true, true>;
template class dependent_factory_t<long, long, double, true, true>;

template class dependent_factory_t<long, long, float, true, false>;
template class dependent_factory_t<long, long, double, true, false>;

template class dependent_factory_t<long, long, float, false, true>;
template class dependent_factory_t<long, long, double, false, true>;

template class dependent_factory_t<long, long, float, false, false>;
template class dependent_factory_t<long, long, double, false, false>;

// Either use EIDir or specialization, can't have both;
// Prefer specialization when EIdir's are not enough
// because of cascaded-dispatcher exhaustive instantiations
// In this case EIDir above are enough;
}  // namespace visitors
}  // namespace cugraph

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

#include <cugraph/visitors/cascaded_dispatch.hpp>
#include <cugraph/visitors/graph_envelope.hpp>

namespace cugraph {
namespace visitors {

// call cascaded dispatcher with factory and erased_pack_t
//
graph_envelope_t::graph_envelope_t(DTypes vertex_tid,
                                   DTypes edge_tid,
                                   DTypes weight_tid,
                                   bool st,
                                   bool mg,
                                   GTypes graph_tid,
                                   erased_pack_t& ep)
  : p_impl_fact_(vertex_dispatcher(vertex_tid, edge_tid, weight_tid, st, mg, graph_tid, ep))
{
}

template class graph_factory_t<graph_t<int, int, float, true, true>>;
template class graph_factory_t<graph_t<int, int, double, true, true>>;

template class graph_factory_t<graph_t<int, int, float, true, false>>;
template class graph_factory_t<graph_t<int, int, double, true, false>>;

template class graph_factory_t<graph_t<int, int, float, false, true>>;
template class graph_factory_t<graph_t<int, int, double, false, true>>;

template class graph_factory_t<graph_t<int, int, float, false, false>>;
template class graph_factory_t<graph_t<int, int, double, false, false>>;

}  // namespace visitors
}  // namespace cugraph

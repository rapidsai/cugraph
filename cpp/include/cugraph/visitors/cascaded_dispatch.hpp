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

#include <array>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "enum_mapping.hpp"
#include "graph_enum_mapping.hpp"

#include "graph_factory.hpp"
#include <cugraph/utilities/graph_traits.hpp>

namespace cugraph {
namespace visitors {

using pair_uniques_t = graph_envelope_t::pair_uniques_t;

// dummy-out non-candidate paths:
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool tr,
          bool mg,
          std::enable_if_t<!is_candidate<vertex_t, edge_t, weight_t>::value, void*> = nullptr>
constexpr pair_uniques_t graph_dispatcher(GTypes graph_type, erased_pack_t& ep)
{
  /// return nullptr;
  return pair_uniques_t{nullptr, nullptr};
}

// final step of cascading: calls factory on erased pack:
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool tr,
          bool mg,
          std::enable_if_t<is_candidate<vertex_t, edge_t, weight_t>::value, void*> = nullptr>
constexpr pair_uniques_t graph_dispatcher(GTypes graph_type, erased_pack_t& ep)
{
  switch (graph_type) {
    case GTypes::GRAPH_T: {
      using graph_t = typename GMapType<vertex_t, edge_t, weight_t, tr, mg, GTypes::GRAPH_T>::type;
      graph_factory_t<graph_t> factory;

      pair_uniques_t p_uniques =
        std::make_pair(factory.make_graph(ep),
                       std::make_unique<dependent_factory_t<vertex_t, edge_t, weight_t, tr, mg>>());

      return p_uniques;
    } break;

    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(graph_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// multi_gpu bool dispatcher:
// resolves bool `multi_gpu`
// and using template arguments vertex_t, edge_t, weight_t, store_transpose
// cascades into next level
// graph_dispatcher()
//
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
constexpr decltype(auto) multi_gpu_dispatcher(bool multi_gpu, GTypes graph_type, erased_pack_t& ep)
{
  switch (multi_gpu) {
    case true: {
      return graph_dispatcher<vertex_t, edge_t, weight_t, store_transposed, true>(graph_type, ep);
    } break;
    case false: {
      return graph_dispatcher<vertex_t, edge_t, weight_t, store_transposed, false>(graph_type, ep);
    }
  }
}

// transpose bool dispatcher:
// resolves bool `store_transpose`
// and using template arguments vertex_t, edge_t, weight_t
// cascades into next level
// multi_gpu_dispatcher()
//
template <typename vertex_t, typename edge_t, typename weight_t>
constexpr decltype(auto) transp_dispatcher(bool store_transposed,
                                           bool multi_gpu,
                                           GTypes graph_type,
                                           erased_pack_t& ep)
{
  switch (store_transposed) {
    case true: {
      return multi_gpu_dispatcher<vertex_t, edge_t, weight_t, true>(multi_gpu, graph_type, ep);
    } break;
    case false: {
      return multi_gpu_dispatcher<vertex_t, edge_t, weight_t, false>(multi_gpu, graph_type, ep);
    }
  }
}

// weight type dispatcher:
// resolves weigth_t from weight_type enum
// and using template arguments vertex_t, edge_t
// cascades into next level
// transp_dispatcher()
//
template <typename vertex_t, typename edge_t>
constexpr decltype(auto) weight_dispatcher(
  DTypes weight_type, bool store_transposed, bool multi_gpu, GTypes graph_type, erased_pack_t& ep)
{
  switch (weight_type) {
    case DTypes::INT32: {
      using weight_t = typename DMapType<DTypes::INT32>::type;
      return transp_dispatcher<vertex_t, edge_t, weight_t>(
        store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::INT64: {
      using weight_t = typename DMapType<DTypes::INT64>::type;
      return transp_dispatcher<vertex_t, edge_t, weight_t>(
        store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::FLOAT32: {
      using weight_t = typename DMapType<DTypes::FLOAT32>::type;
      return transp_dispatcher<vertex_t, edge_t, weight_t>(
        store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::FLOAT64: {
      using weight_t = typename DMapType<DTypes::FLOAT64>::type;
      return transp_dispatcher<vertex_t, edge_t, weight_t>(
        store_transposed, multi_gpu, graph_type, ep);
    } break;
    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(weight_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// edge type dispatcher:
// resolves edge_t from edge_type enum
// and using template argument vertex_t
// cascades into the next level
// weight_dispatcher();
//
template <typename vertex_t>
constexpr decltype(auto) edge_dispatcher(DTypes edge_type,
                                         DTypes weight_type,
                                         bool store_transposed,
                                         bool multi_gpu,
                                         GTypes graph_type,
                                         erased_pack_t& ep)
{
  switch (edge_type) {
    case DTypes::INT32: {
      using edge_t = typename DMapType<DTypes::INT32>::type;
      return weight_dispatcher<vertex_t, edge_t>(
        weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::INT64: {
      using edge_t = typename DMapType<DTypes::INT64>::type;
      return weight_dispatcher<vertex_t, edge_t>(
        weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::FLOAT32: {
      using edge_t = typename DMapType<DTypes::FLOAT32>::type;
      return weight_dispatcher<vertex_t, edge_t>(
        weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::FLOAT64: {
      using edge_t = typename DMapType<DTypes::FLOAT64>::type;
      return weight_dispatcher<vertex_t, edge_t>(
        weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(edge_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// vertex type dispatcher:
// entry point,
// resolves vertex_t from vertex_type enum
// and  cascades into the next level
// edge_dispatcher();
//
inline decltype(auto) vertex_dispatcher(DTypes vertex_type,
                                        DTypes edge_type,
                                        DTypes weight_type,
                                        bool store_transposed,
                                        bool multi_gpu,
                                        GTypes graph_type,
                                        erased_pack_t& ep)
{
  switch (vertex_type) {
    case DTypes::INT32: {
      using vertex_t = typename DMapType<DTypes::INT32>::type;
      return edge_dispatcher<vertex_t>(
        edge_type, weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::INT64: {
      using vertex_t = typename DMapType<DTypes::INT64>::type;
      return edge_dispatcher<vertex_t>(
        edge_type, weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::FLOAT32: {
      using vertex_t = typename DMapType<DTypes::FLOAT32>::type;
      return edge_dispatcher<vertex_t>(
        edge_type, weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    case DTypes::FLOAT64: {
      using vertex_t = typename DMapType<DTypes::FLOAT64>::type;
      return edge_dispatcher<vertex_t>(
        edge_type, weight_type, store_transposed, multi_gpu, graph_type, ep);
    } break;
    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(vertex_type);
      throw std::runtime_error(ss.str());
    }
  }
}

}  // namespace visitors
}  // namespace cugraph

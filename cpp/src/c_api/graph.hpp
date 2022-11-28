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
#pragma once

#include <c_api/array.hpp>
#include <c_api/error.hpp>
#include <cugraph_c/graph.h>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <memory>

namespace cugraph {
namespace c_api {

struct cugraph_graph_t {
  data_type_id_t vertex_type_;
  data_type_id_t edge_type_;
  data_type_id_t weight_type_;
  data_type_id_t edge_type_id_type_;
  bool store_transposed_;
  bool multi_gpu_;

  void* graph_;            // graph_t<...>*
  void* number_map_;       // rmm::device_uvector<vertex_t>*
  void* edge_weights_;     // edge_property_t<
                           //    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                           //    weight_t>*
  void* edge_properties_;  // edge_property_t<
                           //    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                           //    thrust::tuple<edge_t, edge_type_id_t>>>
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
cugraph_error_code_t transpose_storage(raft::handle_t const& handle,
                                       cugraph_graph_t* graph,
                                       cugraph_error_t* error)
{
  if (store_transposed == graph->store_transposed_) {
    if (graph->edge_properties_ != nullptr) {
      error->error_message_ =
        "transpose failed, transposing a graph with edge ID, type pairs unimplemented.";
      return CUGRAPH_NOT_IMPLEMENTED;
    }

    auto p_graph =
      reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
        graph->graph_);

    auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph->number_map_);

    auto optional_edge_weights = std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>(
      std::nullopt);

    if (graph->edge_weights_ != nullptr) {
      auto edge_weights = reinterpret_cast<
        edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>*>(
        graph->edge_weights_);
      optional_edge_weights = std::make_optional(std::move(*edge_weights));
      delete edge_weights;
    }

    auto graph_transposed =
      new cugraph::graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>(handle);

    std::optional<rmm::device_uvector<vertex_t>> new_number_map{std::nullopt};

    auto new_optional_edge_weights = std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, !store_transposed, multi_gpu>, weight_t>>(
      std::nullopt);

    std::tie(*graph_transposed, new_optional_edge_weights, new_number_map) =
      cugraph::transpose_graph_storage(
        handle,
        std::move(*p_graph),
        std::move(optional_edge_weights),
        std::make_optional<rmm::device_uvector<vertex_t>>(std::move(*number_map)));

    *number_map = std::move(new_number_map.value());

    delete p_graph;

    if (new_optional_edge_weights) {
      auto new_edge_weights = new cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
        weight_t>(handle);

      *new_edge_weights    = std::move(new_optional_edge_weights.value());
      graph->edge_weights_ = new_edge_weights;
    }

    graph->graph_            = graph_transposed;
    graph->store_transposed_ = !store_transposed;

    return CUGRAPH_SUCCESS;
  } else {
    error->error_message_ = "transpose failed, value of transpose does not match graph";
    return CUGRAPH_INVALID_INPUT;
  }
}

}  // namespace c_api
}  // namespace cugraph

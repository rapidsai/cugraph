/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "c_api/abstract_functor.hpp"
#include "c_api/core_result.hpp"
#include "c_api/graph.hpp"
#include "c_api/induced_subgraph_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct decompress_to_edgelist_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};

  cugraph::c_api::cugraph_core_result_t const* core_result_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_induced_subgraph_result_t* result_{};

  decompress_to_edgelist_functor(cugraph_resource_handle_t const* handle,
                                 cugraph_graph_t* graph,
                                 bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS)
          ;
      }
      // FIXME: Transpose_storage may have a bug, since if store_transposed is True it can reverse
      // the bool value of is_symmetric
      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        weight_t>*>(graph_->edge_weights_);

      auto edge_ids = reinterpret_cast<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        edge_t>*>(graph_->edge_ids_);

      auto edge_types = reinterpret_cast<cugraph::edge_property_t<
        cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
        edge_type_type_t>*>(graph_->edge_types_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto [result_src, result_dst, result_wgt, result_edge_id, result_edge_type] =
        cugraph::decompress_to_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_type_type_t,
                                        store_transposed,
                                        multi_gpu>(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
          (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
          (number_map != nullptr) ? std::make_optional<raft::device_span<vertex_t const>>(
                                      number_map->data(), number_map->size())
                                  : std::nullopt,
          do_expensive_check_);

      result_ = new cugraph::c_api::cugraph_induced_subgraph_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_dst, graph_->vertex_type_),
        result_wgt ? new cugraph::c_api::cugraph_type_erased_device_array_t(*result_wgt,
                                                                            graph_->weight_type_)
                   : NULL,
        result_edge_id ? new cugraph::c_api::cugraph_type_erased_device_array_t(*result_edge_id,
                                                                                graph_->edge_type_)
                       : NULL,
        result_edge_type ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                             *result_edge_type, graph_->edge_type_id_type_)
                         : NULL,
        NULL};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_decompress_to_edgelist(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  bool_t do_expensive_check,
  cugraph_induced_subgraph_result_t** result,
  cugraph_error_t** error)
{
  decompress_to_edgelist_functor functor(handle, graph, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
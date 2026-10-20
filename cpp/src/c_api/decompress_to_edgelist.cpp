/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/core_result.hpp"
#include "c_api/edgelist.hpp"
#include "c_api/graph.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct decompress_to_edgelist_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};

  cugraph::c_api::cugraph_core_result_t const* core_result_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_edgelist_t* result_{};

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
            typename edge_type_t,
            typename time_stamp_t,
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
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto edge_ids =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_t>*>(graph_->edge_ids_);

      auto edge_types =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_type_t>*>(graph_->edge_types_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      std::vector<cugraph::edge_arithmetic_property_view_t<edge_t>> edge_property_views{};
      if (edge_weights != nullptr) { edge_property_views.push_back(edge_weights->view()); }
      if (edge_ids != nullptr) { edge_property_views.push_back(edge_ids->view()); }
      if (edge_types != nullptr) { edge_property_views.push_back(edge_types->view()); }

      auto [result_src, result_dst, edge_property_values] =
        cugraph::decompress_to_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
          handle_,
          graph_view,
          edge_property_views,
          (number_map != nullptr)
            ? std::make_optional(cugraph::vertex_property_view_t<vertex_t, vertex_t const*>(
                number_map->data(), static_cast<vertex_t>(number_map->size())))
            : std::nullopt,
          std::nullopt,
          do_expensive_check_);

      result_ = new cugraph::c_api::cugraph_edgelist_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_dst, graph_->vertex_type_),
        (edge_weights != nullptr)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(
              std::get<rmm::device_uvector<weight_t>>(edge_property_values[0]),
              graph_->weight_type_)
          : NULL,
        (edge_ids != nullptr) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                  std::get<rmm::device_uvector<edge_t>>(
                                    edge_property_values[(edge_weights != nullptr) ? 1 : 0]),
                                  graph_->edge_type_)
                              : NULL,
        (edge_types != nullptr)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(
              std::get<rmm::device_uvector<edge_type_t>>(edge_property_values.back()),
              graph_->edge_type_id_type_)
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
  cugraph_edgelist_t** result,
  cugraph_error_t** error)
{
  decompress_to_edgelist_functor functor(handle, graph, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

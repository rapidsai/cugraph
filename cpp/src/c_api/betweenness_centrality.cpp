/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cugraph_c/algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/centrality_result.hpp>
#include <c_api/graph.hpp>
#include <c_api/random.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct betweenness_centrality_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertex_list_{};
  bool_t normalized_{};
  bool_t include_endpoints_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_centrality_result_t* result_{};

  betweenness_centrality_functor(cugraph_resource_handle_t const* handle,
                                 cugraph_graph_t* graph,
                                 cugraph_type_erased_device_array_view_t const* vertex_list,
                                 bool_t normalized,
                                 bool_t include_endpoints,
                                 bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertex_list_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        vertex_list)),
      normalized_(normalized),
      include_endpoints_(include_endpoints),
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
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // Betweenness Centrality expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> local_vertices(0, handle_.get_stream());

      std::optional<raft::device_span<vertex_t const>> vertex_span{std::nullopt};
      if (vertex_list_ != nullptr) {
        local_vertices.resize(vertex_list_->size_, handle_.get_stream());
        raft::copy(local_vertices.data(),
                   vertex_list_->as_type<vertex_t>(),
                   vertex_list_->size_,
                   handle_.get_stream());

        if constexpr (multi_gpu) {
          local_vertices =
            cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
              handle_, std::move(local_vertices));
        }

        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          local_vertices.data(),
          local_vertices.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);

        vertex_span =
          raft::device_span<vertex_t const>{local_vertices.data(), local_vertices.size()};
      }

      auto centralities = cugraph::betweenness_centrality<vertex_t, edge_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        vertex_span,
        normalized_,
        include_endpoints_,
        do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_centrality_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(centralities, graph_->weight_type_)};
    }
  }
};

struct edge_betweenness_centrality_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertex_list_{};
  bool_t normalized_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_edge_centrality_result_t* result_{};

  edge_betweenness_centrality_functor(cugraph_resource_handle_t const* handle,
                                      cugraph_graph_t* graph,
                                      cugraph_type_erased_device_array_view_t const* vertex_list,
                                      bool_t normalized,
                                      bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertex_list_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        vertex_list)),
      normalized_(normalized),
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
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // Betweenness Centrality expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto edge_ids = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 edge_t>*>(graph_->edge_ids_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> local_vertices(0, handle_.get_stream());

      std::optional<raft::device_span<vertex_t const>> vertex_span{std::nullopt};
      if (vertex_list_ != nullptr) {
        local_vertices.resize(vertex_list_->size_, handle_.get_stream());
        raft::copy(local_vertices.data(),
                   vertex_list_->as_type<vertex_t>(),
                   vertex_list_->size_,
                   handle_.get_stream());

        if constexpr (multi_gpu) {
          local_vertices =
            cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
              handle_, std::move(local_vertices));
        }

        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          local_vertices.data(),
          local_vertices.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);

        vertex_span =
          raft::device_span<vertex_t const>{local_vertices.data(), local_vertices.size()};
      }

      auto centralities =
        cugraph::edge_betweenness_centrality<vertex_t, edge_t, weight_t, multi_gpu>(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          vertex_span,
          normalized_,
          do_expensive_check_);

      auto [src_ids, dst_ids, output_centralities, output_edge_ids] =
        cugraph::decompress_to_edgelist(
          handle_,
          graph_view,
          std::make_optional(centralities.view()),
          (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
          (number_map != nullptr) ? std::make_optional(raft::device_span<vertex_t const>{
                                      number_map->data(), number_map->size()})
                                  : std::nullopt);

      result_ = new cugraph::c_api::cugraph_edge_centrality_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(src_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(dst_ids, graph_->vertex_type_),
        output_edge_ids ? new cugraph::c_api::cugraph_type_erased_device_array_t(*output_edge_ids,
                                                                                 graph_->edge_type_)
                        : nullptr,
        new cugraph::c_api::cugraph_type_erased_device_array_t(*output_centralities,
                                                               graph_->weight_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_betweenness_centrality(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* vertex_list,
  bool_t normalized,
  bool_t include_endpoints,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error)
{
  betweenness_centrality_functor functor(
    handle, graph, vertex_list, normalized, include_endpoints, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_edge_betweenness_centrality(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* vertex_list,
  bool_t normalized,
  bool_t do_expensive_check,
  cugraph_edge_centrality_result_t** result,
  cugraph_error_t** error)
{
  edge_betweenness_centrality_functor functor(
    handle, graph, vertex_list, normalized, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

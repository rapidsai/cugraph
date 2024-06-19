/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "c_api/lookup_src_dst.hpp"

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/hierarchical_clustering_result.hpp"
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/community_algorithms.h>
#include <cugraph_c/lookup_src_dst.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>

#include <optional>

namespace {

struct build_lookup_map_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::lookup_container_t* result_{};

  build_lookup_map_functor(::cugraph_resource_handle_t const* handle, ::cugraph_graph_t* graph)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph))
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
      // ecg expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_ids = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 edge_t>*>(graph_->edge_type_);
      auto edge_types = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 edge_type_type_t>*>(graph_->edge_type_id_type_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> clusters(0, handle_.get_stream());

      auto lookup_container = new cugraph::lookup_container_t<edge_t, edge_type_type_t, vertex_t>();

      cugraph::build_edge_id_and_type_to_src_dst_lookup_map(
        handle_, graph_view, edge_ids->view(), edge_types->view());

      *lookup_container = std::move(cugraph::build_edge_id_and_type_to_src_dst_lookup_map(
        handle_, graph_view, edge_ids->view(), edge_types->view()));

      auto result = new cugraph::c_api::lookup_container_t{
        graph_->edge_type_, graph_->edge_type_id_type_, graph_->vertex_type_, lookup_container};

      result_ = reinterpret_cast<cugraph::c_api::lookup_container_t*>(result);
    }
  }
};

struct lookup_using_edge_ids_of_single_type_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::lookup_container_t const* lookup_container_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_to_lookup_{nullptr};
  int edge_type_to_lookup_{};
  cugraph::c_api::lookup_result_t* result_{nullptr};

  lookup_using_edge_ids_of_single_type_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_graph_t* graph,
    lookup_container_t const* lookup_container,
    cugraph_type_erased_device_array_view_t const* edge_ids_to_lookup,
    int edge_type_to_lookup)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      lookup_container_(
        reinterpret_cast<cugraph::c_api::lookup_container_t const*>(lookup_container)),
      edge_ids_to_lookup_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          edge_ids_to_lookup)),
      edge_type_to_lookup_(edge_type_to_lookup)
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
      // ecg expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      assert(edge_ids_to_lookup_);

      auto result = cugraph::lookup_endpoints_from_edge_ids_and_single_type<vertex_t,
                                                                            edge_t,
                                                                            edge_type_type_t,
                                                                            multi_gpu>(
        handle_,
        *(reinterpret_cast<cugraph::lookup_container_t<edge_t, edge_type_type_t, vertex_t>*>(
          lookup_container_->lookup_container_)),
        raft::device_span<edge_t const>(edge_ids_to_lookup_->as_type<edge_t>(),
                                        edge_ids_to_lookup_->size_),
        edge_type_to_lookup_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      result_ = new cugraph::c_api::lookup_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(std::get<0>(result),
                                                               lookup_container_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(std::get<1>(result),
                                                               lookup_container_->vertex_type_)};
    }
  }
};

struct lookup_using_edge_ids_and_types_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::lookup_container_t const* lookup_container_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_to_lookup_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_types_to_lookup_{nullptr};
  cugraph::c_api::lookup_result_t* result_{nullptr};

  lookup_using_edge_ids_and_types_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_graph_t* graph,
    lookup_container_t const* lookup_container,
    cugraph_type_erased_device_array_view_t const* edge_ids_to_lookup,
    cugraph_type_erased_device_array_view_t const* edge_types_to_lookup)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      lookup_container_(
        reinterpret_cast<cugraph::c_api::lookup_container_t const*>(lookup_container)),
      edge_ids_to_lookup_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          edge_ids_to_lookup)),
      edge_types_to_lookup_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          edge_types_to_lookup))
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
      // ecg expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      assert(edge_ids_to_lookup_);
      assert(edge_types_to_lookup_);

      auto result = cugraph::
        lookup_endpoints_from_edge_ids_and_types<vertex_t, edge_t, edge_type_type_t, multi_gpu>(
          handle_,
          *(reinterpret_cast<cugraph::lookup_container_t<edge_t, edge_type_type_t, vertex_t>*>(
            lookup_container_->lookup_container_)),
          raft::device_span<edge_t const>(edge_ids_to_lookup_->as_type<edge_t>(),
                                          edge_ids_to_lookup_->size_),
          raft::device_span<edge_type_type_t const>(
            edge_types_to_lookup_->as_type<edge_type_type_t>(), edge_types_to_lookup_->size_));

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      result_ = new cugraph::c_api::lookup_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(std::get<0>(result),
                                                               lookup_container_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(std::get<1>(result),
                                                               lookup_container_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_build_edge_id_and_type_to_src_dst_lookup_map(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  lookup_container_t** lookup_container,
  cugraph_error_t** error)
{
  build_lookup_map_functor functor(handle, graph);

  return cugraph::c_api::run_algorithm(graph, functor, lookup_container, error);
}

extern "C" cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_types(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  const cugraph_type_erased_device_array_view_t* edge_types_to_lookup,
  lookup_result_t** result,
  cugraph_error_t** error)
{
  lookup_using_edge_ids_and_types_functor functor(
    handle, graph, lookup_container, edge_ids_to_lookup, edge_types_to_lookup);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_single_type(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  int edge_type_to_lookup,
  lookup_result_t** result,
  cugraph_error_t** error)
{
  lookup_using_edge_ids_of_single_type_functor functor(
    handle, graph, lookup_container, edge_ids_to_lookup, edge_type_to_lookup);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

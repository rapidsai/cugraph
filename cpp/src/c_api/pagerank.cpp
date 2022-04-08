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

#include <cugraph_c/algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/centrality_result.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct pagerank_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const*
    precomputed_vertex_out_weight_sums_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t* personalization_vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* personalization_values_{};
  double alpha_{};
  double epsilon_{};
  size_t max_iterations_{};
  bool has_initial_guess_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_centrality_result_t* result_{};

  pagerank_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_graph_t* graph,
    cugraph_type_erased_device_array_view_t const* precomputed_vertex_out_weight_sums,
    cugraph_type_erased_device_array_view_t* personalization_vertices,
    cugraph_type_erased_device_array_view_t const* personalization_values,
    double alpha,
    double epsilon,
    size_t max_iterations,
    bool has_initial_guess,
    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      precomputed_vertex_out_weight_sums_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          precomputed_vertex_out_weight_sums)),
      personalization_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(
          personalization_vertices)),
      personalization_values_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          personalization_values)),
      alpha_(alpha),
      epsilon_(epsilon),
      max_iterations_(max_iterations),
      has_initial_guess_(has_initial_guess),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // Pagerank expects store_transposed == true
      if constexpr (!store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph = reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, true, multi_gpu>*>(
        graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<weight_t> pageranks(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());

      if (personalization_vertices_ != nullptr) {
        //
        // Need to renumber personalization_vertices
        //
        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          personalization_vertices_->as_type<vertex_t>(),
          personalization_vertices_->size_,
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);
      }

      cugraph::pagerank<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        precomputed_vertex_out_weight_sums_
          ? std::make_optional(precomputed_vertex_out_weight_sums_->as_type<weight_t const>())
          : std::nullopt,
        personalization_vertices_
          ? std::make_optional(personalization_vertices_->as_type<vertex_t const>())
          : std::nullopt,
        personalization_values_
          ? std::make_optional(personalization_values_->as_type<weight_t const>())
          : std::nullopt,
        personalization_vertices_
          ? std::make_optional(static_cast<vertex_t>(personalization_vertices_->size_))
          : std::nullopt,
        pageranks.data(),
        static_cast<weight_t>(alpha_),
        static_cast<weight_t>(epsilon_),
        max_iterations_,
        has_initial_guess_,
        do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_centrality_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(pageranks, graph_->weight_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t has_initial_guess,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error)
{
  pagerank_functor functor(handle,
                           graph,
                           precomputed_vertex_out_weight_sums,
                           nullptr,
                           nullptr,
                           alpha,
                           epsilon,
                           max_iterations,
                           has_initial_guess,
                           do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_personalized_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
  cugraph_type_erased_device_array_view_t* personalization_vertices,
  const cugraph_type_erased_device_array_view_t* personalization_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t has_initial_guess,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error)
{
  pagerank_functor functor(handle,
                           graph,
                           precomputed_vertex_out_weight_sums,
                           personalization_vertices,
                           personalization_values,
                           alpha,
                           epsilon,
                           max_iterations,
                           has_initial_guess,
                           do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

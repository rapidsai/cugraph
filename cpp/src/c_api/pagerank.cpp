/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct pagerank_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const*
    precomputed_vertex_out_weight_vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const*
    precomputed_vertex_out_weight_sums_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* initial_guess_vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* initial_guess_values_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* personalization_vertices_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* personalization_values_{};
  double alpha_{};
  double epsilon_{};
  size_t max_iterations_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_centrality_result_t* result_{};

  pagerank_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_graph_t* graph,
    cugraph_type_erased_device_array_view_t const* precomputed_vertex_out_weight_vertices,
    cugraph_type_erased_device_array_view_t const* precomputed_vertex_out_weight_sums,
    cugraph_type_erased_device_array_view_t const* initial_guess_vertices,
    cugraph_type_erased_device_array_view_t const* initial_guess_values,
    cugraph_type_erased_device_array_view_t const* personalization_vertices,
    cugraph_type_erased_device_array_view_t const* personalization_values,
    double alpha,
    double epsilon,
    size_t max_iterations,
    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      precomputed_vertex_out_weight_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          precomputed_vertex_out_weight_vertices)),
      precomputed_vertex_out_weight_sums_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          precomputed_vertex_out_weight_sums)),
      initial_guess_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          initial_guess_vertices)),
      initial_guess_values_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          initial_guess_values)),
      personalization_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          personalization_vertices)),
      personalization_values_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          personalization_values)),
      alpha_(alpha),
      epsilon_(epsilon),
      max_iterations_(max_iterations),
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
      // Pagerank expects store_transposed == true
      if constexpr (!store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, true, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<weight_t> pageranks(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());

      rmm::device_uvector<vertex_t> personalization_vertices(0, handle_.get_stream());
      rmm::device_uvector<weight_t> personalization_values(0, handle_.get_stream());

      if (personalization_vertices_ != nullptr) {
        personalization_vertices.resize(personalization_vertices_->size_, handle_.get_stream());
        personalization_values.resize(personalization_values_->size_, handle_.get_stream());

        raft::copy(personalization_vertices.data(),
                   personalization_vertices_->as_type<vertex_t>(),
                   personalization_vertices_->size_,
                   handle_.get_stream());
        raft::copy(personalization_values.data(),
                   personalization_values_->as_type<weight_t>(),
                   personalization_values_->size_,
                   handle_.get_stream());

        if constexpr (multi_gpu) {
          std::tie(personalization_vertices, personalization_values) =
            cugraph::detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
              handle_, std::move(personalization_vertices), std::move(personalization_values));
        }
        //
        // Need to renumber personalization_vertices
        //
        cugraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          personalization_vertices.data(),
          personalization_vertices.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);
      }

      rmm::device_uvector<weight_t> precomputed_vertex_out_weight_sums(0, handle_.get_stream());
      if (precomputed_vertex_out_weight_sums_ != nullptr) {
        rmm::device_uvector<vertex_t> precomputed_vertex_out_weight_vertices(
          precomputed_vertex_out_weight_vertices_->size_, handle_.get_stream());
        precomputed_vertex_out_weight_sums.resize(precomputed_vertex_out_weight_sums_->size_,
                                                  handle_.get_stream());

        raft::copy(precomputed_vertex_out_weight_vertices.data(),
                   precomputed_vertex_out_weight_vertices_->as_type<vertex_t>(),
                   precomputed_vertex_out_weight_vertices_->size_,
                   handle_.get_stream());
        raft::copy(precomputed_vertex_out_weight_sums.data(),
                   precomputed_vertex_out_weight_sums_->as_type<weight_t>(),
                   precomputed_vertex_out_weight_sums_->size_,
                   handle_.get_stream());

        precomputed_vertex_out_weight_sums = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, weight_t, multi_gpu>(
            handle_,
            std::move(precomputed_vertex_out_weight_vertices),
            std::move(precomputed_vertex_out_weight_sums),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            weight_t{0},
            do_expensive_check_);
      }

      if (initial_guess_values_ != nullptr) {
        rmm::device_uvector<vertex_t> initial_guess_vertices(initial_guess_vertices_->size_,
                                                             handle_.get_stream());
        rmm::device_uvector<weight_t> initial_guess_values(initial_guess_values_->size_,
                                                           handle_.get_stream());

        raft::copy(initial_guess_vertices.data(),
                   initial_guess_vertices_->as_type<vertex_t>(),
                   initial_guess_vertices.size(),
                   handle_.get_stream());

        raft::copy(initial_guess_values.data(),
                   initial_guess_values_->as_type<weight_t>(),
                   initial_guess_values.size(),
                   handle_.get_stream());

        pageranks = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, weight_t, multi_gpu>(
            handle_,
            std::move(initial_guess_vertices),
            std::move(initial_guess_values),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            weight_t{0},
            do_expensive_check_);
      }

      cugraph::pagerank<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        precomputed_vertex_out_weight_sums_
          ? std::make_optional(precomputed_vertex_out_weight_sums.data())
          : std::nullopt,
        personalization_vertices_ ? std::make_optional(personalization_vertices.data())
                                  : std::nullopt,
        personalization_values_ ? std::make_optional(personalization_values.data()) : std::nullopt,
        personalization_vertices_
          ? std::make_optional(static_cast<vertex_t>(personalization_vertices.size()))
          : std::nullopt,
        pageranks.data(),
        static_cast<weight_t>(alpha_),
        static_cast<weight_t>(epsilon_),
        max_iterations_,
        initial_guess_values_ != nullptr,
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
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
  const cugraph_type_erased_device_array_view_t* initial_guess_vertices,
  const cugraph_type_erased_device_array_view_t* initial_guess_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error)
{
  if (precomputed_vertex_out_weight_vertices != nullptr) {
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     precomputed_vertex_out_weight_vertices)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and precomputed_vertex_out_weight_vertices must match",
                 *error);
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->weight_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     precomputed_vertex_out_weight_sums)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and precomputed_vertex_out_weight_sums must match",
                 *error);
  }
  if (initial_guess_vertices != nullptr) {
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     initial_guess_vertices)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and initial_guess_vertices must match",
                 *error);
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->weight_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     initial_guess_values)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and initial_guess_values must match",
                 *error);
  }
  pagerank_functor functor(handle,
                           graph,
                           precomputed_vertex_out_weight_vertices,
                           precomputed_vertex_out_weight_sums,
                           initial_guess_vertices,
                           initial_guess_values,
                           nullptr,
                           nullptr,
                           alpha,
                           epsilon,
                           max_iterations,
                           do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_personalized_pagerank(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_vertices,
  const cugraph_type_erased_device_array_view_t* precomputed_vertex_out_weight_sums,
  const cugraph_type_erased_device_array_view_t* initial_guess_vertices,
  const cugraph_type_erased_device_array_view_t* initial_guess_values,
  const cugraph_type_erased_device_array_view_t* personalization_vertices,
  const cugraph_type_erased_device_array_view_t* personalization_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error)
{
  if (precomputed_vertex_out_weight_vertices != nullptr) {
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     precomputed_vertex_out_weight_vertices)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and precomputed_vertex_out_weight_vertices must match",
                 *error);
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->weight_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     precomputed_vertex_out_weight_sums)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and precomputed_vertex_out_weight_sums must match",
                 *error);
  }
  if (initial_guess_vertices != nullptr) {
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     initial_guess_vertices)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and initial_guess_vertices must match",
                 *error);
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->weight_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     initial_guess_values)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and initial_guess_values must match",
                 *error);
  }
  if (personalization_vertices != nullptr) {
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     personalization_vertices)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and personalization_vector must match",
                 *error);
    CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->weight_type_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     personalization_values)
                     ->type_,
                 CUGRAPH_INVALID_INPUT,
                 "vertex type of graph and personalization_vector must match",
                 *error);
  }

  pagerank_functor functor(handle,
                           graph,
                           precomputed_vertex_out_weight_vertices,
                           precomputed_vertex_out_weight_sums,
                           initial_guess_vertices,
                           initial_guess_values,
                           personalization_vertices,
                           personalization_values,
                           alpha,
                           epsilon,
                           max_iterations,
                           do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

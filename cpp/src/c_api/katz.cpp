/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

struct katz_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* betas_{};
  double alpha_{};
  double beta_{};
  double epsilon_{};
  size_t max_iterations_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_centrality_result_t* result_{};

  katz_functor(cugraph_resource_handle_t const* handle,
               cugraph_graph_t* graph,
               cugraph_type_erased_device_array_view_t const* betas,
               double alpha,
               double beta,
               double epsilon,
               size_t max_iterations,
               bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      betas_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(betas)),
      alpha_(alpha),
      beta_(beta),
      epsilon_(epsilon),
      max_iterations_(max_iterations),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // katz expects store_transposed == true
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

      rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(),
                                                 handle_.get_stream());
      rmm::device_uvector<weight_t> betas(0, handle_.get_stream());

      if (betas_ != nullptr) {
        rmm::device_uvector<vertex_t> betas_vertex_ids(
          graph_view.local_vertex_partition_range_size(), handle_.get_stream());
        cugraph::detail::sequence_fill(handle_.get_stream(),
                                       betas_vertex_ids.data(),
                                       betas_vertex_ids.size(),
                                       graph_view.local_vertex_partition_range_first());

        betas.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());

        raft::copy(betas.data(), betas_->as_type<weight_t>(), betas.size(), handle_.get_stream());

        betas = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, weight_t, multi_gpu>(
            handle_,
            std::move(betas_vertex_ids),
            std::move(betas),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            weight_t{0},
            do_expensive_check_);
      }

      cugraph::katz_centrality<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        betas_ == nullptr ? nullptr : betas.data(),
        centralities.data(),
        static_cast<weight_t>(alpha_),
        static_cast<weight_t>(beta_),
        static_cast<weight_t>(epsilon_),
        max_iterations_,
        false,
        true,
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

}  // namespace

extern "C" cugraph_error_code_t cugraph_katz_centrality(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* betas,
  double alpha,
  double beta,
  double epsilon,
  size_t max_iterations,
  bool_t do_expensive_check,
  cugraph_centrality_result_t** result,
  cugraph_error_t** error)
{
  katz_functor functor(
    handle, graph, nullptr, alpha, beta, epsilon, max_iterations, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

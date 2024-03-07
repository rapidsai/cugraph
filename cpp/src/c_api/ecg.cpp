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

#include <cugraph_c/community_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/graph_helper.hpp>
#include <c_api/hierarchical_clustering_result.hpp>
#include <c_api/random.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <optional>

namespace {

struct ecg_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  double min_weight_{0.1};
  size_t ensemble_size_{10};
  size_t max_level_{0};
  double threshold_{0.001};
  double resolution_{1};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_hierarchical_clustering_result_t* result_{};

  ecg_functor(::cugraph_resource_handle_t const* handle,
              ::cugraph_rng_state_t* rng_state,
              ::cugraph_graph_t* graph,
              double min_weight,
              size_t ensemble_size,
              size_t max_level,
              double threshold,
              double resolution,
              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      max_level_(max_level),
      threshold_(threshold),
      resolution_(resolution),
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

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> clusters(0, handle_.get_stream());

      weight_t modularity;

      std::tie(clusters, std::ignore, modularity) =
        cugraph::ecg(handle_,
                     rng_state_->rng_state_,
                     graph_view,
                     (edge_weights != nullptr)
                       ? std::make_optional(edge_weights->view())
                       : std::make_optional(cugraph::c_api::create_constant_edge_property(
                                              handle_, graph_view, weight_t{1})
                                              .view()),
                     static_cast<weight_t>(min_weight_),
                     ensemble_size_,
                     max_level_,
                     static_cast<weight_t>(threshold_),
                     static_cast<weight_t>(resolution_));

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_hierarchical_clustering_result_t{
        modularity,
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(clusters, graph_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_ecg(const cugraph_resource_handle_t* handle,
                                            cugraph_rng_state_t* rng_state,
                                            cugraph_graph_t* graph,
                                            double min_weight,
                                            size_t ensemble_size,
                                            size_t max_level,
                                            double threshold,
                                            double resolution,
                                            bool_t do_expensive_check,
                                            cugraph_hierarchical_clustering_result_t** result,
                                            cugraph_error_t** error)
{
  ecg_functor functor(handle,
                      rng_state,
                      graph,
                      min_weight,
                      ensemble_size,
                      max_level,
                      threshold,
                      resolution,
                      do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
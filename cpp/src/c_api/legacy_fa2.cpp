/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "c_api/capi_helper.hpp"
#include "c_api/graph.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_clustering_result_t {
  cugraph_type_erased_device_array_t* vertices_{nullptr};
  cugraph_type_erased_device_array_t* clusters_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct force_atlas2_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  //cugraph::c_api::cugraph_type_erased_device_array_view_t* pos_{};
  const int max_iter_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t* x_start_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t* y_start_{};
  bool outbound_attraction_distribution_{};
  bool lin_log_mode_{};
  bool prevent_overlapping_{};
  const double edge_weight_influence_{};
  const double jitter_tolerance_{};
  bool barnes_hut_optimize_{};
  const double barnes_hut_theta_{};
  const double scaling_ratio_{};
  bool strong_gravity_mode_{};
  const double gravity_{};
  bool verbose_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_type_erased_device_array_t* result_{};;

  force_atlas2_functor(::cugraph_resource_handle_t const* handle,
                       ::cugraph_graph_t* graph,
                       //::cugraph_type_erased_device_array_view_t* pos,
                       const int max_iter,
                       ::cugraph_type_erased_device_array_view_t* x_start,
                       ::cugraph_type_erased_device_array_view_t* y_start,
                       bool outbound_attraction_distribution,
                       bool lin_log_mode,
                       bool prevent_overlapping,
                       const double edge_weight_influence,
                       const double jitter_tolerance,
                       bool barnes_hut_optimize,
                       const double barnes_hut_theta,
                       const double scaling_ratio,
                       bool strong_gravity_mode,
                       const double gravity,
                       bool verbose,
                       bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      //pos_(
      //  reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(pos)),
      max_iter_(max_iter),
      x_start_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(x_start)),
      y_start_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(y_start)),
      outbound_attraction_distribution_(outbound_attraction_distribution),
      lin_log_mode_(lin_log_mode),
      prevent_overlapping_(prevent_overlapping),
      edge_weight_influence_(edge_weight_influence),
      jitter_tolerance_(jitter_tolerance),
      barnes_hut_optimize_(barnes_hut_optimize),
      barnes_hut_theta_(barnes_hut_theta),
      scaling_ratio_(scaling_ratio),
      strong_gravity_mode_(strong_gravity_mode),
      gravity_(gravity),
      verbose_(verbose),
      // Simply pass x and y axis result pointer and populate it, similar to 'einterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(clusters))'
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
    } else if constexpr (multi_gpu) {
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // balanced_cut expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      // Decompress to edgelist

      auto [srcs, dsts, wgts, dummy_0, dummy_1] =
        cugraph::decompress_to_edgelist<vertex_t, edge_t, weight_t, int32_t, false, multi_gpu>(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          std::nullopt,
          std::nullopt,
          (number_map != nullptr) ? std::make_optional(raft::device_span<vertex_t const>{
                                      number_map->data(), number_map->size()})
                                  : std::nullopt);

      cugraph::legacy::GraphCOOView<vertex_t, edge_t, weight_t> legacy_coo_graph_view(
        const_cast<vertex_t*>(srcs.data()),
        const_cast<vertex_t*>(dsts.data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(wgts->data()),
        edge_partition_view.offsets().size() - 1, //FIXME: instead call graph->number_of_vertices
        edge_partition_view.indices().size()); //FIXME: instead call edge_partition_view.number_of_edges()    

      cugraph::internals::GraphBasedDimRedCallback* callback = nullptr;
      
      //cugraph::legacy::GraphBasedDimRedCallback()

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());


      //#if 0

      rmm::device_uvector<float> pos(2 * (edge_partition_view.offsets().size() - 1),
                                       handle_.get_stream());

      cugraph::force_atlas2<vertex_t, edge_t, weight_t>(
          handle_,
          legacy_coo_graph_view,
          pos.data(),
          //pos_->as_type<float>(), // pos is an empty array of size num_vertices * 2
          max_iter_,
          x_start_->as_type<float>(),
          y_start_->as_type<float>(),
          outbound_attraction_distribution_,
          lin_log_mode_,
          prevent_overlapping_,
          edge_weight_influence_,
          jitter_tolerance_,
          barnes_hut_optimize_,
          barnes_hut_theta_,
          scaling_ratio_,
          strong_gravity_mode_,
          gravity_,
          verbose_,
          callback
          );
      
      result_ = new cugraph::c_api::cugraph_type_erased_device_array_t(pos, graph_->weight_type_);

    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_force_atlas2(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph_type_erased_device_array_t** pos,
  const int max_iter,
  cugraph_type_erased_device_array_view_t* x_start,
  cugraph_type_erased_device_array_view_t* y_start,
  bool_t outbound_attraction_distribution,
  bool_t lin_log_mode,
  bool_t prevent_overlapping,
  const double edge_weight_influence,
  const double jitter_tolerance,
  bool_t barnes_hut_optimize,
  const double barnes_hut_theta,
  const double scaling_ratio,
  bool_t strong_gravity_mode,
  const double gravity,
  bool_t verbose,
  bool_t do_expensive_check,
  cugraph_error_t** error)
{
  force_atlas2_functor functor(handle,
                               graph,
                               //pos,
                               max_iter,
                               x_start,
                               y_start,
                               outbound_attraction_distribution,
                               lin_log_mode,
                               prevent_overlapping,
                               edge_weight_influence,
                               jitter_tolerance,
                               barnes_hut_optimize,
                               barnes_hut_theta,
                               scaling_ratio,
                               strong_gravity_mode,
                               gravity,
                               verbose,
                               do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, pos, error);
}


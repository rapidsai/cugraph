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
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_layout_result_t {
  cugraph_type_erased_device_array_t* vertices_{nullptr};
  cugraph_type_erased_device_array_t* x_axis_{nullptr};
  cugraph_type_erased_device_array_t* y_axis_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct force_atlas2_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  int max_iter_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t* x_start_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t* y_start_{};
  bool outbound_attraction_distribution_{};
  bool lin_log_mode_{};
  bool prevent_overlapping_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t* vertex_radius_{};
  double overlap_scaling_ratio_{};
  double edge_weight_influence_{};
  double jitter_tolerance_{};
  bool barnes_hut_optimize_{};
  double barnes_hut_theta_{};
  double scaling_ratio_{};
  bool strong_gravity_mode_{};
  double gravity_{};
  bool verbose_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_layout_result_t* result_{};
  ;

  force_atlas2_functor(::cugraph_resource_handle_t const* handle,
                       cugraph_rng_state_t* rng_state,
                       ::cugraph_graph_t* graph,
                       int max_iter,
                       ::cugraph_type_erased_device_array_view_t* x_start,
                       ::cugraph_type_erased_device_array_view_t* y_start,
                       bool outbound_attraction_distribution,
                       bool lin_log_mode,
                       bool prevent_overlapping,
                       ::cugraph_type_erased_device_array_view_t* vertex_radius,
                       double overlap_scaling_ratio,
                       double edge_weight_influence,
                       double jitter_tolerance,
                       bool barnes_hut_optimize,
                       double barnes_hut_theta,
                       double scaling_ratio,
                       bool strong_gravity_mode,
                       double gravity,
                       bool verbose,
                       bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      max_iter_(max_iter),
      x_start_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(x_start)),
      y_start_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(y_start)),
      outbound_attraction_distribution_(outbound_attraction_distribution),
      lin_log_mode_(lin_log_mode),
      prevent_overlapping_(prevent_overlapping),
      vertex_radius_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(vertex_radius)),
      overlap_scaling_ratio_(overlap_scaling_ratio),
      edge_weight_influence_(edge_weight_influence),
      jitter_tolerance_(jitter_tolerance),
      barnes_hut_optimize_(barnes_hut_optimize),
      barnes_hut_theta_(barnes_hut_theta),
      scaling_ratio_(scaling_ratio),
      strong_gravity_mode_(strong_gravity_mode),
      gravity_(gravity),
      verbose_(verbose),
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

      auto [srcs, dsts, wgts, edge_ids, edge_types] =
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
        (edge_weights == nullptr) ? tmp_weights.data() : const_cast<weight_t*>(wgts->data()),
        graph->number_of_vertices(),
        edge_partition_view.number_of_edges());

      cugraph::internals::GraphBasedDimRedCallback* callback = nullptr;

      rmm::device_uvector<float> pos(2 * (edge_partition_view.offsets().size() - 1),
                                     handle_.get_stream());

      std::optional<rmm::device_uvector<vertex_t>> cp_number_map{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> number_map_pos{std::nullopt};

      if (x_start_ != nullptr) {
        // re-order x_start and y_start based on internal vertex IDs
        cp_number_map = rmm::device_uvector<vertex_t>{number_map->size(), handle_.get_stream()};

        raft::copy(
          cp_number_map->data(), number_map->data(), number_map->size(), handle_.get_stream());

        number_map_pos = rmm::device_uvector<vertex_t>{number_map->size(), handle_.get_stream()};

        cugraph::detail::sequence_fill(
          handle_.get_stream(), number_map_pos->begin(), number_map_pos->size(), vertex_t{0});

        cugraph::c_api::detail::sort_by_key(
          handle_,
          raft::device_span<vertex_t>{cp_number_map->data(), cp_number_map->size()},
          raft::device_span<vertex_t>{number_map_pos->data(), number_map_pos->size()});

        cugraph::c_api::detail::sort_tuple_by_key(
          handle_,
          raft::device_span<vertex_t>{number_map_pos->data(), number_map_pos->size()},
          std::make_tuple(raft::device_span<float>{x_start_->as_type<float>(), x_start_->size_},
                          raft::device_span<float>{y_start_->as_type<float>(), y_start_->size_}));
      }

      cugraph::force_atlas2<vertex_t, edge_t, weight_t>(
        handle_,
        // rng_state_->rng_state_, # FIXME: Add support
        legacy_coo_graph_view,
        pos.data(),
        max_iter_,
        x_start_ != nullptr ? x_start_->as_type<float>() : nullptr,
        y_start_ != nullptr ? y_start_->as_type<float>() : nullptr,
        outbound_attraction_distribution_,
        lin_log_mode_,
        prevent_overlapping_,
        vertex_radius_,
        overlap_scaling_ratio_,
        edge_weight_influence_,
        jitter_tolerance_,
        barnes_hut_optimize_,
        barnes_hut_theta_,
        scaling_ratio_,
        strong_gravity_mode_,
        gravity_,
        verbose_,
        callback);

      rmm::device_uvector<float> x_axis(graph_view.local_vertex_partition_range_size(),
                                        handle_.get_stream());

      rmm::device_uvector<float> y_axis(graph_view.local_vertex_partition_range_size(),
                                        handle_.get_stream());

      raft::copy(x_axis.data(), pos.data(), x_axis.size(), handle_.get_stream());

      raft::copy(y_axis.data(), pos.data() + x_axis.size(), x_axis.size(), handle_.get_stream());

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_layout_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(x_axis,
                                                               cugraph_data_type_id_t::FLOAT32),
        new cugraph::c_api::cugraph_type_erased_device_array_t(y_axis,
                                                               cugraph_data_type_id_t::FLOAT32)};
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_vertices(
  cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_x_axis(
  cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->x_axis_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_y_axis(
  cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->y_axis_->view());
}

extern "C" void cugraph_layout_result_free(cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->x_axis_;
  delete internal_pointer->y_axis_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_force_atlas2(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  int max_iter,
  cugraph_type_erased_device_array_view_t* x_start,
  cugraph_type_erased_device_array_view_t* y_start,
  bool_t outbound_attraction_distribution,
  bool_t lin_log_mode,
  bool_t prevent_overlapping,
  cugraph_type_erased_device_array_view_t* vertex_radius,
  double overlap_scaling_ratio,
  double edge_weight_influence,
  double jitter_tolerance,
  bool_t barnes_hut_optimize,
  double barnes_hut_theta,
  double scaling_ratio,
  bool_t strong_gravity_mode,
  double gravity,
  bool_t verbose,
  bool_t do_expensive_check,
  cugraph::c_api::cugraph_layout_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(((x_start == nullptr) && (y_start == nullptr)) ||
                 ((x_start != nullptr) && (y_start != nullptr)),
               CUGRAPH_INVALID_INPUT,
               "Both x_start and y_start should either be NULL or specified.",
               *error);

  if (x_start != nullptr) {
    CAPI_EXPECTS(
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(x_start)
          ->type_ ==
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(y_start)
          ->type_,
      CUGRAPH_INVALID_INPUT,
      "Both x_start and y_start type  must match when provided",
      *error);
  }

  force_atlas2_functor functor(handle,
                               rng_state,
                               graph,
                               max_iter,
                               x_start,
                               y_start,
                               outbound_attraction_distribution,
                               lin_log_mode,
                               prevent_overlapping,
                               vertex_radius,
                               overlap_scaling_ratio,
                               edge_weight_influence,
                               jitter_tolerance,
                               barnes_hut_optimize,
                               barnes_hut_theta,
                               scaling_ratio,
                               strong_gravity_mode,
                               gravity,
                               verbose,
                               do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

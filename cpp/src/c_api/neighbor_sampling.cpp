/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "c_api/graph.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/properties.hpp"
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"
#include "sampling/detail/sampling_utils.hpp"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/sampling_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_sampling_options_t {
  bool_t with_replacement_{FALSE};
  bool_t return_hops_{FALSE};
  prior_sources_behavior_t prior_sources_behavior_{prior_sources_behavior_t::DEFAULT};
  bool_t dedupe_sources_{FALSE};
  bool_t renumber_results_{FALSE};
  cugraph_compression_type_t compression_type_{cugraph_compression_type_t::COO};
  bool_t compress_per_hop_{FALSE};
  bool_t retain_seeds_{FALSE};
};

struct sampling_flags_t {
  prior_sources_behavior_t prior_sources_behavior_{prior_sources_behavior_t::DEFAULT};
  bool_t return_hops_{FALSE};
  bool_t dedupe_sources_{FALSE};
  bool_t with_replacement_{FALSE};
};

struct cugraph_sample_result_t {
  cugraph_type_erased_device_array_t* major_offsets_{nullptr};
  cugraph_type_erased_device_array_t* majors_{nullptr};
  cugraph_type_erased_device_array_t* minors_{nullptr};
  cugraph_type_erased_device_array_t* edge_id_{nullptr};
  cugraph_type_erased_device_array_t* edge_type_{nullptr};
  cugraph_type_erased_device_array_t* wgt_{nullptr};
  cugraph_type_erased_device_array_t* hop_{nullptr};
  cugraph_type_erased_device_array_t* label_hop_offsets_{nullptr};
  cugraph_type_erased_device_array_t* label_type_hop_offsets_{nullptr};
  cugraph_type_erased_device_array_t* label_{nullptr};
  cugraph_type_erased_device_array_t* renumber_map_{nullptr};
  cugraph_type_erased_device_array_t* renumber_map_offsets_{nullptr};
  cugraph_type_erased_device_array_t* edge_renumber_map_{nullptr};
  cugraph_type_erased_device_array_t* edge_renumber_map_offsets_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

// Deprecated functor
struct uniform_neighbor_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertex_labels_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_list_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_to_comm_rank_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_offsets_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_sampling_options_t options_{};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  uniform_neighbor_sampling_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_graph_t* graph,
    cugraph_type_erased_device_array_view_t const* start_vertices,
    cugraph_type_erased_device_array_view_t const* start_vertex_labels,
    cugraph_type_erased_device_array_view_t const* label_list,
    cugraph_type_erased_device_array_view_t const* label_to_comm_rank,
    cugraph_type_erased_device_array_view_t const* label_offsets,
    cugraph_type_erased_host_array_view_t const* fan_out,
    cugraph_rng_state_t* rng_state,
    cugraph::c_api::cugraph_sampling_options_t options,
    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
      start_vertex_labels_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertex_labels)),
      label_list_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        label_list)),
      label_to_comm_rank_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          label_to_comm_rank)),
      label_offsets_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          label_offsets)),
      fan_out_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      options_(options),
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
    using label_t = int32_t;

    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // uniform_nbr_sample expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto edge_ids =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_t>*>(graph_->edge_ids_);

      auto edge_types =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_type_t>*>(graph_->edge_types_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_, handle_.get_stream());
      raft::copy(start_vertices.data(),
                 start_vertices_->as_type<vertex_t>(),
                 start_vertices.size(),
                 handle_.get_stream());

      std::optional<rmm::device_uvector<label_t>> start_vertex_labels{std::nullopt};

      if (start_vertex_labels_ != nullptr) {
        start_vertex_labels =
          rmm::device_uvector<label_t>{start_vertex_labels_->size_, handle_.get_stream()};
        raft::copy(start_vertex_labels->data(),
                   start_vertex_labels_->as_type<label_t>(),
                   start_vertex_labels_->size_,
                   handle_.get_stream());
      }

      if constexpr (multi_gpu) {
        if (start_vertex_labels) {
          std::tie(start_vertices, *start_vertex_labels) = cugraph::shuffle_ext_vertex_value_pairs(
            handle_, std::move(start_vertices), std::move(*start_vertex_labels));
        } else {
          start_vertices = cugraph::shuffle_ext_vertices(handle_, std::move(start_vertices));
        }
      }

      //
      // Need to renumber start_vertices
      //
      cugraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start_vertices.data(),
        start_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        do_expensive_check_);

      auto&& [src, dst, wgt, edge_id, edge_type, hop, edge_label, offsets] =
        cugraph::uniform_neighbor_sample(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
          (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
          raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
          (start_vertex_labels_ != nullptr)
            ? std::make_optional<raft::device_span<label_t const>>(start_vertex_labels->data(),
                                                                   start_vertex_labels->size())
            : std::nullopt,
          (label_list_ != nullptr)
            ? std::make_optional(std::make_tuple(
                raft::device_span<label_t const>{label_list_->as_type<label_t>(),
                                                 label_list_->size_},
                raft::device_span<label_t const>{label_to_comm_rank_->as_type<label_t>(),
                                                 label_to_comm_rank_->size_}))
            : std::nullopt,
          raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
          rng_state_->rng_state_,
          options_.return_hops_,
          options_.with_replacement_,
          options_.prior_sources_behavior_,
          options_.dedupe_sources_,
          do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        src.data(),
        src.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        dst.data(),
        dst.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      std::optional<rmm::device_uvector<vertex_t>> majors{std::nullopt};
      rmm::device_uvector<vertex_t> minors(0, handle_.get_stream());
      std::optional<rmm::device_uvector<size_t>> major_offsets{std::nullopt};

      std::optional<rmm::device_uvector<size_t>> label_hop_offsets{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
      std::optional<rmm::device_uvector<size_t>> renumber_map_offsets{std::nullopt};

      bool src_is_major = (options_.compression_type_ == cugraph_compression_type_t::CSR) ||
                          (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                          (options_.compression_type_ == cugraph_compression_type_t::COO);

      if (options_.renumber_results_) {
        if (options_.compression_type_ == cugraph_compression_type_t::COO) {
          // COO

          rmm::device_uvector<vertex_t> output_majors(0, handle_.get_stream());
          rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
          std::tie(output_majors,
                   minors,
                   wgt,
                   edge_id,
                   edge_type,
                   label_hop_offsets,
                   output_renumber_map,
                   renumber_map_offsets) =
            cugraph::renumber_and_sort_sampled_edgelist<vertex_t>(
              handle_,
              std::move(src),
              std::move(dst),
              std::move(wgt),
              std::move(edge_id),
              std::move(edge_type),
              std::move(hop),
              options_.retain_seeds_
                ? std::make_optional(raft::device_span<vertex_t const>{
                    start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                : std::nullopt,
              options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                         label_offsets_->as_type<size_t>(), label_offsets_->size_})
                                     : std::nullopt,
              offsets ? std::make_optional(
                          raft::device_span<size_t const>{offsets->data(), offsets->size()})
                      : std::nullopt,
              edge_label ? edge_label->size() : size_t{1},
              hop ? fan_out_->size_ : size_t{1},
              src_is_major,
              do_expensive_check_);

          majors.emplace(std::move(output_majors));
          renumber_map.emplace(std::move(output_renumber_map));
        } else {
          // (D)CSC, (D)CSR

          bool doubly_compress = (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                                 (options_.compression_type_ == cugraph_compression_type_t::DCSC);

          rmm::device_uvector<size_t> output_major_offsets(0, handle_.get_stream());
          rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
          std::tie(majors,
                   output_major_offsets,
                   minors,
                   wgt,
                   edge_id,
                   edge_type,
                   label_hop_offsets,
                   output_renumber_map,
                   renumber_map_offsets) =
            cugraph::renumber_and_compress_sampled_edgelist<vertex_t>(
              handle_,
              std::move(src),
              std::move(dst),
              std::move(wgt),
              std::move(edge_id),
              std::move(edge_type),
              std::move(hop),
              options_.retain_seeds_
                ? std::make_optional(raft::device_span<vertex_t const>{
                    start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                : std::nullopt,
              options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                         label_offsets_->as_type<size_t>(), label_offsets_->size_})
                                     : std::nullopt,
              offsets ? std::make_optional(
                          raft::device_span<size_t const>{offsets->data(), offsets->size()})
                      : std::nullopt,
              edge_label ? edge_label->size() : size_t{1},
              hop ? fan_out_->size_ : size_t{1},
              src_is_major,
              options_.compress_per_hop_,
              doubly_compress,
              do_expensive_check_);

          renumber_map.emplace(std::move(output_renumber_map));
          major_offsets.emplace(std::move(output_major_offsets));
        }

        // These are now represented by label_hop_offsets
        hop.reset();
        offsets.reset();
      } else {
        if (options_.compression_type_ != cugraph_compression_type_t::COO) {
          CUGRAPH_FAIL("Can only use COO format if not renumbering");
        }

        std::tie(src, dst, wgt, edge_id, edge_type, label_hop_offsets) =
          cugraph::sort_sampled_edgelist(handle_,
                                         std::move(src),
                                         std::move(dst),
                                         std::move(wgt),
                                         std::move(edge_id),
                                         std::move(edge_type),
                                         std::move(hop),
                                         offsets
                                           ? std::make_optional(raft::device_span<size_t const>{
                                               offsets->data(), offsets->size()})
                                           : std::nullopt,
                                         edge_label ? edge_label->size() : size_t{1},
                                         hop ? fan_out_->size_ : size_t{1},
                                         src_is_major,
                                         do_expensive_check_);

        majors.emplace(std::move(src));
        minors = std::move(dst);

        hop.reset();
        offsets.reset();
      }

      result_ = new cugraph::c_api::cugraph_sample_result_t{
        (major_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*major_offsets, SIZE_T)
          : nullptr,
        (majors)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*majors, graph_->vertex_type_)
          : nullptr,
        new cugraph::c_api::cugraph_type_erased_device_array_t(minors, graph_->vertex_type_),
        (edge_id)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edge_id, graph_->edge_type_)
          : nullptr,
        (edge_type) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                        *edge_type, graph_->edge_type_id_type_)
                    : nullptr,
        (wgt) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*wgt, graph_->weight_type_)
              : nullptr,
        (hop) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*hop, INT32)
              : nullptr,  // FIXME get rid of this
        (label_hop_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*label_hop_offsets, SIZE_T)
          : nullptr,
        nullptr,
        (edge_label)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(edge_label.value(), INT32)
          : nullptr,
        (renumber_map) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                           renumber_map.value(), graph_->vertex_type_)
                       : nullptr,
        (renumber_map_offsets) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                   renumber_map_offsets.value(), SIZE_T)
                               : nullptr,
        nullptr,
        nullptr};
    }
  }
};

// Deprecated functor
struct biased_neighbor_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_edge_property_view_t const* edge_biases_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertex_labels_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_list_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_to_comm_rank_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_offsets_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_sampling_options_t options_{};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  biased_neighbor_sampling_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_graph_t* graph,
    cugraph_edge_property_view_t const* edge_biases,
    cugraph_type_erased_device_array_view_t const* start_vertices,
    cugraph_type_erased_device_array_view_t const* start_vertex_labels,
    cugraph_type_erased_device_array_view_t const* label_list,
    cugraph_type_erased_device_array_view_t const* label_to_comm_rank,
    cugraph_type_erased_device_array_view_t const* label_offsets,
    cugraph_type_erased_host_array_view_t const* fan_out,
    cugraph_rng_state_t* rng_state,
    cugraph::c_api::cugraph_sampling_options_t options,
    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      edge_biases_(
        reinterpret_cast<cugraph::c_api::cugraph_edge_property_view_t const*>(edge_biases)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
      start_vertex_labels_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertex_labels)),
      label_list_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        label_list)),
      label_to_comm_rank_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          label_to_comm_rank)),
      label_offsets_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          label_offsets)),
      fan_out_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      options_(options),
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
    using label_t = int32_t;

    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // uniform_nbr_sample expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto edge_ids =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_t>*>(graph_->edge_ids_);

      auto edge_types =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_type_t>*>(graph_->edge_types_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto edge_biases =
        edge_biases_ ? reinterpret_cast<cugraph::edge_property_view_t<edge_t, weight_t const*>*>(
                         edge_biases_->edge_property_)
                     : nullptr;

      rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_, handle_.get_stream());
      raft::copy(start_vertices.data(),
                 start_vertices_->as_type<vertex_t>(),
                 start_vertices.size(),
                 handle_.get_stream());

      std::optional<rmm::device_uvector<label_t>> start_vertex_labels{std::nullopt};

      if (start_vertex_labels_ != nullptr) {
        start_vertex_labels =
          rmm::device_uvector<label_t>{start_vertex_labels_->size_, handle_.get_stream()};
        raft::copy(start_vertex_labels->data(),
                   start_vertex_labels_->as_type<label_t>(),
                   start_vertex_labels_->size_,
                   handle_.get_stream());
      }

      if constexpr (multi_gpu) {
        if (start_vertex_labels) {
          std::tie(start_vertices, *start_vertex_labels) = cugraph::shuffle_ext_vertex_value_pairs(
            handle_, std::move(start_vertices), std::move(*start_vertex_labels));
        } else {
          start_vertices = cugraph::shuffle_ext_vertices(handle_, std::move(start_vertices));
        }
      }

      //
      // Need to renumber start_vertices
      //
      cugraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start_vertices.data(),
        start_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        do_expensive_check_);

      auto&& [src, dst, wgt, edge_id, edge_type, hop, edge_label, offsets] =
        cugraph::biased_neighbor_sample(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
          (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
          (edge_biases != nullptr) ? *edge_biases : edge_weights->view(),
          raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
          (start_vertex_labels_ != nullptr)
            ? std::make_optional<raft::device_span<label_t const>>(start_vertex_labels->data(),
                                                                   start_vertex_labels->size())
            : std::nullopt,
          (label_list_ != nullptr)
            ? std::make_optional(std::make_tuple(
                raft::device_span<label_t const>{label_list_->as_type<label_t>(),
                                                 label_list_->size_},
                raft::device_span<label_t const>{label_to_comm_rank_->as_type<label_t>(),
                                                 label_to_comm_rank_->size_}))
            : std::nullopt,
          raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
          rng_state_->rng_state_,
          options_.return_hops_,
          options_.with_replacement_,
          options_.prior_sources_behavior_,
          options_.dedupe_sources_,
          do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        src.data(),
        src.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        dst.data(),
        dst.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      std::optional<rmm::device_uvector<vertex_t>> majors{std::nullopt};
      rmm::device_uvector<vertex_t> minors(0, handle_.get_stream());
      std::optional<rmm::device_uvector<size_t>> major_offsets{std::nullopt};

      std::optional<rmm::device_uvector<size_t>> label_hop_offsets{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
      std::optional<rmm::device_uvector<size_t>> renumber_map_offsets{std::nullopt};

      bool src_is_major = (options_.compression_type_ == cugraph_compression_type_t::CSR) ||
                          (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                          (options_.compression_type_ == cugraph_compression_type_t::COO);

      if (options_.renumber_results_) {
        if (options_.compression_type_ == cugraph_compression_type_t::COO) {
          // COO

          rmm::device_uvector<vertex_t> output_majors(0, handle_.get_stream());
          rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
          std::tie(output_majors,
                   minors,
                   wgt,
                   edge_id,
                   edge_type,
                   label_hop_offsets,
                   output_renumber_map,
                   renumber_map_offsets) =
            cugraph::renumber_and_sort_sampled_edgelist<vertex_t>(
              handle_,
              std::move(src),
              std::move(dst),
              std::move(wgt),
              std::move(edge_id),
              std::move(edge_type),
              std::move(hop),
              options_.retain_seeds_
                ? std::make_optional(raft::device_span<vertex_t const>{
                    start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                : std::nullopt,
              options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                         label_offsets_->as_type<size_t>(), label_offsets_->size_})
                                     : std::nullopt,
              offsets ? std::make_optional(
                          raft::device_span<size_t const>{offsets->data(), offsets->size()})
                      : std::nullopt,
              edge_label ? edge_label->size() : size_t{1},
              hop ? fan_out_->size_ : size_t{1},
              src_is_major,
              do_expensive_check_);

          majors.emplace(std::move(output_majors));
          renumber_map.emplace(std::move(output_renumber_map));
        } else {
          // (D)CSC, (D)CSR

          bool doubly_compress = (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                                 (options_.compression_type_ == cugraph_compression_type_t::DCSC);

          rmm::device_uvector<size_t> output_major_offsets(0, handle_.get_stream());
          rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
          std::tie(majors,
                   output_major_offsets,
                   minors,
                   wgt,
                   edge_id,
                   edge_type,
                   label_hop_offsets,
                   output_renumber_map,
                   renumber_map_offsets) =
            cugraph::renumber_and_compress_sampled_edgelist<vertex_t>(
              handle_,
              std::move(src),
              std::move(dst),
              std::move(wgt),
              std::move(edge_id),
              std::move(edge_type),
              std::move(hop),
              options_.retain_seeds_
                ? std::make_optional(raft::device_span<vertex_t const>{
                    start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                : std::nullopt,
              options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                         label_offsets_->as_type<size_t>(), label_offsets_->size_})
                                     : std::nullopt,
              offsets ? std::make_optional(
                          raft::device_span<size_t const>{offsets->data(), offsets->size()})
                      : std::nullopt,
              edge_label ? edge_label->size() : size_t{1},
              hop ? fan_out_->size_ : size_t{1},
              src_is_major,
              options_.compress_per_hop_,
              doubly_compress,
              do_expensive_check_);

          renumber_map.emplace(std::move(output_renumber_map));
          major_offsets.emplace(std::move(output_major_offsets));
        }

        // These are now represented by label_hop_offsets
        hop.reset();
        offsets.reset();
      } else {
        if (options_.compression_type_ != cugraph_compression_type_t::COO) {
          CUGRAPH_FAIL("Can only use COO format if not renumbering");
        }

        std::tie(src, dst, wgt, edge_id, edge_type, label_hop_offsets) =
          cugraph::sort_sampled_edgelist(handle_,
                                         std::move(src),
                                         std::move(dst),
                                         std::move(wgt),
                                         std::move(edge_id),
                                         std::move(edge_type),
                                         std::move(hop),
                                         offsets
                                           ? std::make_optional(raft::device_span<size_t const>{
                                               offsets->data(), offsets->size()})
                                           : std::nullopt,
                                         edge_label ? edge_label->size() : size_t{1},
                                         hop ? fan_out_->size_ : size_t{1},
                                         src_is_major,
                                         do_expensive_check_);

        majors.emplace(std::move(src));
        minors = std::move(dst);

        hop.reset();
        offsets.reset();
      }

      result_ = new cugraph::c_api::cugraph_sample_result_t{
        (major_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*major_offsets, SIZE_T)
          : nullptr,
        (majors)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*majors, graph_->vertex_type_)
          : nullptr,
        new cugraph::c_api::cugraph_type_erased_device_array_t(minors, graph_->vertex_type_),
        (edge_id)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edge_id, graph_->edge_type_)
          : nullptr,
        (edge_type) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                        *edge_type, graph_->edge_type_id_type_)
                    : nullptr,
        (wgt) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*wgt, graph_->weight_type_)
              : nullptr,
        (hop) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*hop, INT32)
              : nullptr,  // FIXME get rid of this
        (label_hop_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*label_hop_offsets, SIZE_T)
          : nullptr,
        nullptr,
        (edge_label)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(edge_label.value(), INT32)
          : nullptr,
        (renumber_map) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                           renumber_map.value(), graph_->vertex_type_)
                       : nullptr,
        (renumber_map_offsets) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                   renumber_map_offsets.value(), SIZE_T)
                               : nullptr,
        nullptr,
        nullptr};
    }
  }
};

struct neighbor_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_edge_property_view_t const* edge_biases_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* starting_vertex_label_offsets_{
    nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertex_type_offsets_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  int num_edge_types_{};
  cugraph::c_api::cugraph_sampling_options_t options_{};
  bool is_biased_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  neighbor_sampling_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_rng_state_t* rng_state,
    cugraph_graph_t* graph,
    cugraph_edge_property_view_t const* edge_biases,
    cugraph_type_erased_device_array_view_t const* start_vertices,
    cugraph_type_erased_device_array_view_t const* starting_vertex_label_offsets,
    cugraph_type_erased_device_array_view_t const* vertex_type_offsets,
    cugraph_type_erased_host_array_view_t const* fan_out,
    int num_edge_types,
    cugraph::c_api::cugraph_sampling_options_t options,
    bool is_biased,
    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      edge_biases_(
        reinterpret_cast<cugraph::c_api::cugraph_edge_property_view_t const*>(edge_biases)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
      starting_vertex_label_offsets_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          starting_vertex_label_offsets)),
      vertex_type_offsets_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          vertex_type_offsets)),
      fan_out_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)),
      num_edge_types_(num_edge_types),
      options_(options),
      is_biased_(is_biased),
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
    using label_t = int32_t;

    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // uniform_nbr_sample expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto edge_ids =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_t>*>(graph_->edge_ids_);

      auto edge_types =
        reinterpret_cast<cugraph::edge_property_t<edge_t, edge_type_t>*>(graph_->edge_types_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto edge_biases =
        edge_biases_ ? reinterpret_cast<cugraph::edge_property_view_t<edge_t, weight_t const*>*>(
                         edge_biases_->edge_property_)
                     : nullptr;

      rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_, handle_.get_stream());
      raft::copy(start_vertices.data(),
                 start_vertices_->as_type<vertex_t>(),
                 start_vertices.size(),
                 handle_.get_stream());

      std::optional<rmm::device_uvector<label_t>> start_vertex_labels{std::nullopt};
      std::optional<rmm::device_uvector<label_t>> label_to_comm_rank{
        std::nullopt};  // global after allgatherv

      std::optional<rmm::device_uvector<edge_t>> renumbered_and_sorted_edge_id_renumber_map(
        std::nullopt);
      std::optional<rmm::device_uvector<size_t>>
        renumbered_and_sorted_edge_id_renumber_map_label_type_offsets(std::nullopt);

      if (starting_vertex_label_offsets_ != nullptr) {
        // Retrieve the start_vertex_labels
        start_vertex_labels = cugraph::detail::convert_starting_vertex_label_offsets_to_labels(
          handle_,
          raft::device_span<size_t const>{starting_vertex_label_offsets_->as_type<size_t>(),
                                          starting_vertex_label_offsets_->size_});

        // Get the number of labels on each GPU

        if constexpr (multi_gpu) {
          auto num_local_labels = starting_vertex_label_offsets_->size_ - 1;

          auto global_labels = cugraph::host_scalar_allgather(
            handle_.get_comms(), num_local_labels, handle_.get_stream());

          std::exclusive_scan(
            global_labels.begin(), global_labels.end(), global_labels.begin(), label_t{0});

          // Compute the global starting_vertex_label_offsets

          cugraph::detail::transform_increment_ints(
            raft::device_span<label_t>{(*start_vertex_labels).data(),
                                       (*start_vertex_labels).size()},
            (label_t)global_labels[handle_.get_comms().get_rank()],
            handle_.get_stream());

          rmm::device_uvector<label_t> unique_labels((*start_vertex_labels).size(),
                                                     handle_.get_stream());
          raft::copy(unique_labels.data(),
                     (*start_vertex_labels).data(),
                     unique_labels.size(),
                     handle_.get_stream());

          // Get unique labels
          // sort the start_vertex_labels
          cugraph::detail::sort_ints(
            handle_.get_stream(),
            raft::device_span<label_t>{unique_labels.data(), unique_labels.size()});

          auto num_unique_labels = cugraph::detail::unique_ints(
            handle_.get_stream(),
            raft::device_span<label_t>{unique_labels.data(), unique_labels.size()});

          rmm::device_uvector<label_t> local_label_to_comm_rank(num_unique_labels,
                                                                handle_.get_stream());

          cugraph::detail::scalar_fill(
            handle_.get_stream(),
            local_label_to_comm_rank.begin(),  // This should be rename to rank
            local_label_to_comm_rank.size(),
            label_t{handle_.get_comms().get_rank()});

          // Perform allgather to get global_label_to_comm_rank_d_vector
          auto recvcounts = cugraph::host_scalar_allgather(
            handle_.get_comms(), num_unique_labels, handle_.get_stream());

          std::vector<size_t> displacements(recvcounts.size());
          std::exclusive_scan(
            recvcounts.begin(), recvcounts.end(), displacements.begin(), size_t{0});

          label_to_comm_rank = rmm::device_uvector<label_t>(
            displacements.back() + recvcounts.back(), handle_.get_stream());

          cugraph::device_allgatherv(
            handle_.get_comms(),
            local_label_to_comm_rank.begin(),
            (*label_to_comm_rank).begin(),
            raft::host_span<size_t const>(recvcounts.data(), recvcounts.size()),
            raft::host_span<size_t const>(displacements.data(), displacements.size()),
            handle_.get_stream());

          std::tie(start_vertices, *start_vertex_labels) = cugraph::shuffle_ext_vertex_value_pairs(
            handle_, std::move(start_vertices), std::move(*start_vertex_labels));
        }
      } else {
        if constexpr (multi_gpu) {
          start_vertices = cugraph::shuffle_ext_vertices(handle_, std::move(start_vertices));
        }
      }
      //
      // Need to renumber start_vertices
      //
      cugraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start_vertices.data(),
        start_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        do_expensive_check_);

      rmm::device_uvector<vertex_t> src(0, handle_.get_stream());
      rmm::device_uvector<vertex_t> dst(0, handle_.get_stream());
      std::optional<rmm::device_uvector<weight_t>> wgt{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> edge_id{std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> edge_type{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> hop{std::nullopt};
      std::optional<rmm::device_uvector<label_t>> edge_label{std::nullopt};
      std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

      // FIXME: For biased sampling, the user should pass either biases or edge weights,
      // otherwised throw an error and suggest the user to call uniform neighbor sample instead

      if (num_edge_types_ > 1) {
        CUGRAPH_EXPECTS(edge_types != nullptr,
                        "edge types are necessary for heterogeneous sampling.");

        // call heterogeneous neighbor sample
        if (is_biased_) {
          std::tie(src, dst, wgt, edge_id, edge_type, hop, offsets) =
            cugraph::heterogeneous_biased_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              edge_types->view(),
              (edge_biases != nullptr) ? *edge_biases : edge_weights->view(),
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              (starting_vertex_label_offsets_ != nullptr)
                ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                   (*start_vertex_labels).size())
                : std::nullopt,
              label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                     (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                 : std::nullopt,
              raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
              num_edge_types_,
              cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                        options_.return_hops_,
                                        options_.dedupe_sources_,
                                        options_.with_replacement_},
              do_expensive_check_);
        } else {
          std::tie(src, dst, wgt, edge_id, edge_type, hop, offsets) =
            cugraph::heterogeneous_uniform_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              edge_types->view(),
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              (starting_vertex_label_offsets_ != nullptr)
                ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                   (*start_vertex_labels).size())
                : std::nullopt,
              label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                     (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                 : std::nullopt,
              raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
              num_edge_types_,
              cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                        options_.return_hops_,
                                        options_.dedupe_sources_,
                                        options_.with_replacement_},
              do_expensive_check_);
        }
      } else {
        // Call homogeneous neighbor sample
        if (is_biased_) {
          std::tie(src, dst, wgt, edge_id, edge_type, hop, offsets) =
            cugraph::homogeneous_biased_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
              (edge_biases != nullptr) ? *edge_biases : edge_weights->view(),
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              (starting_vertex_label_offsets_ != nullptr)
                ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                   (*start_vertex_labels).size())
                : std::nullopt,
              label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                     (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                 : std::nullopt,
              raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
              cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                        options_.return_hops_,
                                        options_.dedupe_sources_,
                                        options_.with_replacement_},
              do_expensive_check_);
        } else {
          std::tie(src, dst, wgt, edge_id, edge_type, hop, offsets) =
            cugraph::homogeneous_uniform_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              (starting_vertex_label_offsets_ != nullptr)
                ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                   (*start_vertex_labels).size())
                : std::nullopt,
              label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                     (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                 : std::nullopt,
              raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
              cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                        options_.return_hops_,
                                        options_.dedupe_sources_,
                                        options_.with_replacement_},
              do_expensive_check_);
        }
      }

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        src.data(),
        src.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        dst.data(),
        dst.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      std::optional<rmm::device_uvector<vertex_t>> majors{std::nullopt};
      rmm::device_uvector<vertex_t> minors(0, handle_.get_stream());
      std::optional<rmm::device_uvector<size_t>> major_offsets{std::nullopt};

      std::optional<rmm::device_uvector<size_t>> label_hop_offsets{std::nullopt};
      std::optional<rmm::device_uvector<size_t>> label_type_hop_offsets{std::nullopt};

      std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
      std::optional<rmm::device_uvector<size_t>> renumber_map_offsets{std::nullopt};

      bool src_is_major = (options_.compression_type_ == cugraph_compression_type_t::CSR) ||
                          (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                          (options_.compression_type_ == cugraph_compression_type_t::COO);

      // Extract the edge_label from the offsets
      if (offsets) {
        edge_label = cugraph::c_api::expand_sparse_offsets(
          raft::device_span<size_t const>{(*offsets).data(), (*offsets).size()},
          label_t{0},
          handle_.get_stream());
      }

      if (options_.renumber_results_) {
        if (src.size() > 0) {          // Only renumber if there are edgelist to renumber
          if (num_edge_types_ == 1) {  // homogeneous renumbering
            if (options_.compression_type_ == cugraph_compression_type_t::COO) {
              // COO

              rmm::device_uvector<vertex_t> output_majors(0, handle_.get_stream());
              rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
              std::tie(output_majors,
                       minors,
                       wgt,
                       edge_id,
                       edge_type,
                       label_hop_offsets,
                       output_renumber_map,
                       renumber_map_offsets) =
                cugraph::renumber_and_sort_sampled_edgelist<vertex_t>(
                  handle_,
                  std::move(src),
                  std::move(dst),
                  std::move(wgt),
                  std::move(edge_id),
                  std::move(edge_type),
                  std::move(hop),
                  options_.retain_seeds_
                    ? std::make_optional(raft::device_span<vertex_t const>{
                        start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                    : std::nullopt,
                  options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                             starting_vertex_label_offsets_->as_type<size_t>(),
                                             starting_vertex_label_offsets_->size_})
                                         : std::nullopt,
                  offsets ? std::make_optional(
                              raft::device_span<size_t const>{offsets->data(), offsets->size()})
                          : std::nullopt,
                  offsets ? (*offsets).size() - 1 : size_t{1},
                  hop ? fan_out_->size_ : size_t{1},
                  src_is_major,
                  do_expensive_check_);

              majors.emplace(std::move(output_majors));
              renumber_map.emplace(std::move(output_renumber_map));
            } else {
              // (D)CSC, (D)CSR

              bool doubly_compress =
                (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                (options_.compression_type_ == cugraph_compression_type_t::DCSC);

              rmm::device_uvector<size_t> output_major_offsets(0, handle_.get_stream());
              rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());

              std::tie(majors,
                       output_major_offsets,
                       minors,
                       wgt,
                       edge_id,
                       edge_type,
                       label_hop_offsets,
                       output_renumber_map,
                       renumber_map_offsets) =
                cugraph::renumber_and_compress_sampled_edgelist<vertex_t>(
                  handle_,
                  std::move(src),
                  std::move(dst),
                  std::move(wgt),
                  std::move(edge_id),
                  std::move(edge_type),
                  std::move(hop),
                  options_.retain_seeds_
                    ? std::make_optional(raft::device_span<vertex_t const>{
                        start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                    : std::nullopt,
                  options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                             starting_vertex_label_offsets_->as_type<size_t>(),
                                             starting_vertex_label_offsets_->size_})
                                         : std::nullopt,
                  offsets ? std::make_optional(
                              raft::device_span<size_t const>{offsets->data(), offsets->size()})
                          : std::nullopt,
                  edge_label ? (*offsets).size() - 1 : size_t{1},  // FIXME: update edge_label
                  hop ? fan_out_->size_ : size_t{1},
                  src_is_major,
                  options_.compress_per_hop_,
                  doubly_compress,
                  do_expensive_check_);

              renumber_map.emplace(std::move(output_renumber_map));
              major_offsets.emplace(std::move(output_major_offsets));
            }

            // These are now represented by label_hop_offsets
            hop.reset();
            offsets.reset();

          } else {  // heterogeneous renumbering

            rmm::device_uvector<vertex_t> vertex_type_offsets(2, handle_.get_stream());

            if (vertex_type_offsets_ == nullptr) {
              // If no 'vertex_type_offsets' is provided, all vertices are assumed to have
              // a vertex type of value 1.
              cugraph::detail::stride_fill(handle_.get_stream(),
                                           vertex_type_offsets.begin(),
                                           vertex_type_offsets.size(),
                                           vertex_t{0},
                                           vertex_t{graph_view.local_vertex_partition_range_size()}

              );
            }

            rmm::device_uvector<vertex_t> output_majors(0, handle_.get_stream());
            rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());

            std::tie(output_majors,
                     minors,
                     wgt,
                     edge_id,
                     label_type_hop_offsets,  // Contains information about the type and hop offsets
                     output_renumber_map,
                     renumber_map_offsets,
                     renumbered_and_sorted_edge_id_renumber_map,
                     renumbered_and_sorted_edge_id_renumber_map_label_type_offsets) =
              cugraph::heterogeneous_renumber_and_sort_sampled_edgelist<vertex_t>(
                handle_,
                std::move(src),
                std::move(dst),
                std::move(wgt),
                std::move(edge_id),
                std::move(edge_type),
                std::move(hop),
                options_.retain_seeds_
                  ? std::make_optional(raft::device_span<vertex_t const>{
                      start_vertices_->as_type<vertex_t>(), start_vertices_->size_})
                  : std::nullopt,
                options_.retain_seeds_ ? std::make_optional(raft::device_span<size_t const>{
                                           starting_vertex_label_offsets_->as_type<size_t>(),
                                           starting_vertex_label_offsets_->size_})
                                       : std::nullopt,
                offsets ? std::make_optional(
                            raft::device_span<size_t const>{offsets->data(), offsets->size()})
                        : std::nullopt,

                (vertex_type_offsets_ != nullptr)
                  ? raft::device_span<vertex_t const>{vertex_type_offsets_->as_type<vertex_t>(),
                                                      vertex_type_offsets_->size_}
                  : raft::device_span<vertex_t const>{vertex_type_offsets.data(),
                                                      vertex_type_offsets.size()},

                edge_label ? (*offsets).size() - 1 : size_t{1},
                hop ? (((fan_out_->size_ % num_edge_types_) == 0)
                         ? (fan_out_->size_ / num_edge_types_)
                         : ((fan_out_->size_ / num_edge_types_) + 1))
                    : size_t{1},
                (vertex_type_offsets_ != nullptr) ? vertex_type_offsets_->size_ - 1
                                                  : vertex_type_offsets.size() - 1,

                // num_vertex_type is by default 1 if 'vertex_type_offsets' is not provided.
                num_edge_types_,
                src_is_major,
                do_expensive_check_);
            if (edge_type) {
              (*edge_type)
                .resize(raft::device_span<size_t const>{(*label_type_hop_offsets).data(),
                                                        (*label_type_hop_offsets).size()}
                            .back() -
                          1,
                        handle_.get_stream());
              cugraph::detail::sequence_fill(
                handle_.get_stream(), (*edge_type).begin(), (*edge_type).size(), edge_type_t{0});
            }

            majors.emplace(std::move(output_majors));
            // FIXME: Need to update renumber_map because default values are being passed
            renumber_map.emplace(std::move(output_renumber_map));
          }
        }

      } else {
        if (options_.compression_type_ != cugraph_compression_type_t::COO) {
          CUGRAPH_FAIL("Can only use COO format if not renumbering");
        }

        std::tie(src, dst, wgt, edge_id, edge_type, label_hop_offsets) =
          cugraph::sort_sampled_edgelist(handle_,
                                         std::move(src),
                                         std::move(dst),
                                         std::move(wgt),
                                         std::move(edge_id),
                                         std::move(edge_type),
                                         std::move(hop),
                                         offsets
                                           ? std::make_optional(raft::device_span<size_t const>{
                                               offsets->data(), offsets->size()})
                                           : std::nullopt,
                                         // derive label size from offset size instead of performing
                                         // thrust::unique on edge_label.
                                         edge_label ? (*offsets).size() - 1 : size_t{1},
                                         hop ? fan_out_->size_ : size_t{1},
                                         src_is_major,
                                         do_expensive_check_);

        majors.emplace(std::move(src));
        minors = std::move(dst);

        hop.reset();
        offsets.reset();
      }

      result_ = new cugraph::c_api::cugraph_sample_result_t{
        (major_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*major_offsets, SIZE_T)
          : nullptr,
        (majors)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*majors, graph_->vertex_type_)
          : nullptr,
        new cugraph::c_api::cugraph_type_erased_device_array_t(minors, graph_->vertex_type_),
        (edge_id)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edge_id, graph_->edge_type_)
          : nullptr,
        (edge_type) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                        *edge_type, graph_->edge_type_id_type_)
                    : nullptr,
        (wgt) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*wgt, graph_->weight_type_)
              : nullptr,
        (hop) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*hop, INT32)
              : nullptr,  // FIXME get rid of this
        (label_hop_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*label_hop_offsets, SIZE_T)
          : nullptr,
        (label_type_hop_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*label_type_hop_offsets, SIZE_T)
          : nullptr,
        (edge_label)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(edge_label.value(), INT32)
          : nullptr,
        (renumber_map) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                           renumber_map.value(), graph_->vertex_type_)
                       : nullptr,
        (renumber_map_offsets) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                   renumber_map_offsets.value(), SIZE_T)
                               : nullptr,
        (renumbered_and_sorted_edge_id_renumber_map)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(
              renumbered_and_sorted_edge_id_renumber_map.value(), graph_->edge_type_)
          : nullptr,
        (renumbered_and_sorted_edge_id_renumber_map_label_type_offsets)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(
              renumbered_and_sorted_edge_id_renumber_map_label_type_offsets.value(), SIZE_T)
          : nullptr};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_sampling_options_create(
  cugraph_sampling_options_t** options, cugraph_error_t** error)
{
  *options =
    reinterpret_cast<cugraph_sampling_options_t*>(new cugraph::c_api::cugraph_sampling_options_t());
  if (*options == nullptr) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  return CUGRAPH_SUCCESS;
}

extern "C" void cugraph_sampling_set_retain_seeds(cugraph_sampling_options_t* options, bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->retain_seeds_ = value;
}

extern "C" void cugraph_sampling_set_renumber_results(cugraph_sampling_options_t* options,
                                                      bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->renumber_results_ = value;
}

extern "C" void cugraph_sampling_set_compress_per_hop(cugraph_sampling_options_t* options,
                                                      bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->compress_per_hop_ = value;
}

extern "C" void cugraph_sampling_set_with_replacement(cugraph_sampling_options_t* options,
                                                      bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->with_replacement_ = value;
}

extern "C" void cugraph_sampling_set_return_hops(cugraph_sampling_options_t* options, bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->return_hops_ = value;
}

extern "C" void cugraph_sampling_set_compression_type(cugraph_sampling_options_t* options,
                                                      cugraph_compression_type_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  switch (value) {
    case COO: internal_pointer->compression_type_ = cugraph_compression_type_t::COO; break;
    case CSR: internal_pointer->compression_type_ = cugraph_compression_type_t::CSR; break;
    case CSC: internal_pointer->compression_type_ = cugraph_compression_type_t::CSC; break;
    case DCSR: internal_pointer->compression_type_ = cugraph_compression_type_t::DCSR; break;
    case DCSC: internal_pointer->compression_type_ = cugraph_compression_type_t::DCSC; break;
    default: CUGRAPH_FAIL("Invalid compression type");
  }
}

extern "C" void cugraph_sampling_set_prior_sources_behavior(cugraph_sampling_options_t* options,
                                                            cugraph_prior_sources_behavior_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  switch (value) {
    case CARRY_OVER:
      internal_pointer->prior_sources_behavior_ = cugraph::prior_sources_behavior_t::CARRY_OVER;
      break;
    case EXCLUDE:
      internal_pointer->prior_sources_behavior_ = cugraph::prior_sources_behavior_t::EXCLUDE;
      break;
    default:
      internal_pointer->prior_sources_behavior_ = cugraph::prior_sources_behavior_t::DEFAULT;
      break;
  }
}

extern "C" void cugraph_sampling_set_dedupe_sources(cugraph_sampling_options_t* options,
                                                    bool_t value)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  internal_pointer->dedupe_sources_ = value;
}

extern "C" void cugraph_sampling_options_free(cugraph_sampling_options_t* options)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t*>(options);
  delete internal_pointer;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_sources(
  const cugraph_sample_result_t* result)
{
  // Deprecated.
  return cugraph_sample_result_get_majors(result);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_destinations(
  const cugraph_sample_result_t* result)
{
  // Deprecated.
  return cugraph_sample_result_get_minors(result);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_majors(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return (internal_pointer->majors_ != nullptr)
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->majors_->view())

           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_major_offsets(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return (internal_pointer->major_offsets_ != nullptr)
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->major_offsets_->view())

           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_minors(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->minors_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_start_labels(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->label_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_id(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_id_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_id_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_type(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_type_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_type_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_weight(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->wgt_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->wgt_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_hop(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->hop_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->hop_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_label_hop_offsets(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->label_hop_offsets_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_hop_offsets_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_sample_result_get_label_type_hop_offsets(const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->label_type_hop_offsets_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_type_hop_offsets_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_index(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->edge_id_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_offsets(
  const cugraph_sample_result_t* result)
{
  // Deprecated.
  return cugraph_sample_result_get_label_hop_offsets(result);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_renumber_map(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->renumber_map_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_renumber_map_offsets(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->renumber_map_offsets_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_renumber_map(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_renumber_map_->view());
}

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_sample_result_get_edge_renumber_map_offsets(const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->renumber_map_ == nullptr
           ? NULL
           : reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_renumber_map_offsets_->view());
}

extern "C" cugraph_error_code_t cugraph_test_uniform_neighborhood_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* edge_id,
  const cugraph_type_erased_device_array_view_t* edge_type,
  const cugraph_type_erased_device_array_view_t* weight,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;
  size_t n_bytes{0};
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  if (!handle) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  // Create unique_ptrs and release them during cugraph_sample_result_t
  // construction. This allows the arrays to be cleaned up if this function
  // returns early on error.
  using device_array_unique_ptr_t =
    std::unique_ptr<cugraph_type_erased_device_array_t,
                    decltype(&cugraph_type_erased_device_array_free)>;

  // copy srcs to new device array
  cugraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, srcs, &new_device_srcs_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy dsts to new device array
  cugraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, dsts, &new_device_dsts_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy weights to new device array
  cugraph_type_erased_device_array_t* new_device_weight_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, weight, &new_device_weight_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_weight(new_device_weight_ptr,
                                              &cugraph_type_erased_device_array_free);

  // copy edge ids to new device array
  cugraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, edge_id, &new_device_edge_id_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                               &cugraph_type_erased_device_array_free);

  // copy edge types to new device array
  cugraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, edge_type, &new_device_edge_type_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                 &cugraph_type_erased_device_array_free);
  // copy hop ids to new device array
  cugraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, hop, &new_device_hop_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy labels to new device array
  cugraph_type_erased_device_array_t* new_device_label_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, label, &new_device_label_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                             &cugraph_type_erased_device_array_free);

  // create new cugraph_sample_result_t
  *result = reinterpret_cast<cugraph_sample_result_t*>(new cugraph::c_api::cugraph_sample_result_t{
    nullptr,
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_srcs.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_dsts.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_id.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_type.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_weight.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_hop.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_label.release())});

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_test_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* edge_id,
  const cugraph_type_erased_device_array_view_t* edge_type,
  const cugraph_type_erased_device_array_view_t* wgt,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;
  size_t n_bytes{0};
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  if (!handle) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  // Create unique_ptrs and release them during cugraph_sample_result_t
  // construction. This allows the arrays to be cleaned up if this function
  // returns early on error.
  using device_array_unique_ptr_t =
    std::unique_ptr<cugraph_type_erased_device_array_t,
                    decltype(&cugraph_type_erased_device_array_free)>;

  // copy srcs to new device array
  cugraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, srcs, &new_device_srcs_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy dsts to new device array
  cugraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, dsts, &new_device_dsts_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy edge_id to new device array
  cugraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};

  if (edge_id != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, edge_id, &new_device_edge_id_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                               &cugraph_type_erased_device_array_free);

  // copy edge_type to new device array
  cugraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};

  if (edge_type != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, edge_type, &new_device_edge_type_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                 &cugraph_type_erased_device_array_free);

  // copy wgt to new device array
  cugraph_type_erased_device_array_t* new_device_wgt_ptr{nullptr};
  if (wgt != NULL) {
    error_code =
      cugraph_type_erased_device_array_create_from_view(handle, wgt, &new_device_wgt_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_wgt(new_device_wgt_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy hop to new device array
  cugraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, hop, &new_device_hop_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy label to new device array
  cugraph_type_erased_device_array_t* new_device_label_ptr{nullptr};

  if (label != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, label, &new_device_label_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                             &cugraph_type_erased_device_array_free);

  // create new cugraph_sample_result_t
  *result = reinterpret_cast<cugraph_sample_result_t*>(new cugraph::c_api::cugraph_sample_result_t{
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_srcs.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_dsts.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_id.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_type.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_wgt.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_label.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_hop.release())});

  return CUGRAPH_SUCCESS;
}

extern "C" void cugraph_sample_result_free(cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t*>(result);
  delete internal_pointer->major_offsets_;
  delete internal_pointer->majors_;
  delete internal_pointer->minors_;
  delete internal_pointer->edge_id_;
  delete internal_pointer->edge_type_;
  delete internal_pointer->wgt_;
  delete internal_pointer->hop_;
  delete internal_pointer->label_hop_offsets_;
  delete internal_pointer->label_;
  delete internal_pointer->renumber_map_;
  delete internal_pointer->renumber_map_offsets_;
  delete internal_pointer;
}

cugraph_error_code_t cugraph_uniform_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* start_vertex_labels,
  const cugraph_type_erased_device_array_view_t* label_list,
  const cugraph_type_erased_device_array_view_t* label_to_comm_rank,
  const cugraph_type_erased_device_array_view_t* label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  cugraph_rng_state_t* rng_state,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  CAPI_EXPECTS((!options_cpp.retain_seeds_) || (label_offsets != nullptr),
               CUGRAPH_INVALID_INPUT,
               "must specify label_offsets if retain_seeds is true",
               *error);

  CAPI_EXPECTS((start_vertex_labels == nullptr) ||
                 (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                    start_vertex_labels)
                    ->type_ == INT32),
               CUGRAPH_INVALID_INPUT,
               "start_vertex_labels should be of type int",
               *error);

  CAPI_EXPECTS((label_to_comm_rank == nullptr) || (start_vertex_labels != nullptr),
               CUGRAPH_INVALID_INPUT,
               "cannot specify label_to_comm_rank unless start_vertex_labels is also specified",
               *error);

  CAPI_EXPECTS((label_to_comm_rank == nullptr) || (label_list != nullptr),
               CUGRAPH_INVALID_INPUT,
               "cannot specify label_to_comm_rank unless label_list is also specified",
               *error);

  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out should be of type int",
    *error);

  //  Deprecated functor
  uniform_neighbor_sampling_functor functor{handle,
                                            graph,
                                            start_vertices,
                                            start_vertex_labels,
                                            label_list,
                                            label_to_comm_rank,
                                            label_offsets,
                                            fan_out,
                                            rng_state,
                                            std::move(options_cpp),
                                            do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_biased_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* start_vertex_labels,
  const cugraph_type_erased_device_array_view_t* label_list,
  const cugraph_type_erased_device_array_view_t* label_to_comm_rank,
  const cugraph_type_erased_device_array_view_t* label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  cugraph_rng_state_t* rng_state,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  CAPI_EXPECTS(
    (edge_biases != nullptr) ||
      (reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr),
    CUGRAPH_INVALID_INPUT,
    "edge_biases is required if the graph is not weighted",
    *error);

  CAPI_EXPECTS((!options_cpp.retain_seeds_) || (label_offsets != nullptr),
               CUGRAPH_INVALID_INPUT,
               "must specify label_offsets if retain_seeds is true",
               *error);

  CAPI_EXPECTS((start_vertex_labels == nullptr) ||
                 (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                    start_vertex_labels)
                    ->type_ == INT32),
               CUGRAPH_INVALID_INPUT,
               "start_vertex_labels should be of type int",
               *error);

  CAPI_EXPECTS((label_to_comm_rank == nullptr) || (start_vertex_labels != nullptr),
               CUGRAPH_INVALID_INPUT,
               "cannot specify label_to_comm_rank unless start_vertex_labels is also specified",
               *error);

  CAPI_EXPECTS((label_to_comm_rank == nullptr) || (label_list != nullptr),
               CUGRAPH_INVALID_INPUT,
               "cannot specify label_to_comm_rank unless label_list is also specified",
               *error);

  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out should be of type int",
    *error);

  // Deprecated functor
  biased_neighbor_sampling_functor functor{handle,
                                           graph,
                                           edge_biases,
                                           start_vertices,
                                           start_vertex_labels,
                                           label_list,
                                           label_to_comm_rank,
                                           label_offsets,
                                           fan_out,
                                           rng_state,
                                           std::move(options_cpp),
                                           do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_heterogeneous_uniform_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_device_array_view_t* vertex_type_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  int num_edge_types,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  // FIXME: Should we maintain this contition?
  CAPI_EXPECTS((!options_cpp.retain_seeds_) || (starting_vertex_label_offsets != nullptr),
               CUGRAPH_INVALID_INPUT,
               "must specify starting_vertex_label_offsets if retain_seeds is true",
               *error);

  CAPI_EXPECTS((starting_vertex_label_offsets == nullptr) ||
                 (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                    starting_vertex_label_offsets)
                    ->type_ == SIZE_T),
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_label_offsets should be of type size_t",
               *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out should be of type int",
    *error);

  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  neighbor_sampling_functor functor{handle,
                                    rng_state,
                                    graph,
                                    nullptr,
                                    start_vertices,
                                    starting_vertex_label_offsets,
                                    vertex_type_offsets,
                                    fan_out,
                                    num_edge_types,
                                    std::move(options_cpp),
                                    FALSE,
                                    do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_heterogeneous_biased_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_device_array_view_t* vertex_type_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  int num_edge_types,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  CAPI_EXPECTS(
    (edge_biases != nullptr) ||
      (reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr),
    CUGRAPH_INVALID_INPUT,
    "edge_biases is required if the graph is not weighted",
    *error);

  // FIXME: Should we maintain this contition?
  CAPI_EXPECTS((!options_cpp.retain_seeds_) || (starting_vertex_label_offsets != nullptr),
               CUGRAPH_INVALID_INPUT,
               "must specify starting_vertex_label_offsets if retain_seeds is true",
               *error);

  CAPI_EXPECTS((starting_vertex_label_offsets == nullptr) ||
                 (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                    starting_vertex_label_offsets)
                    ->type_ == SIZE_T),
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_label_offsets should be of type size_t",
               *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out should be of type int",
    *error);

  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  neighbor_sampling_functor functor{handle,
                                    rng_state,
                                    graph,
                                    edge_biases,
                                    start_vertices,
                                    starting_vertex_label_offsets,
                                    vertex_type_offsets,
                                    fan_out,
                                    num_edge_types,
                                    std::move(options_cpp),
                                    TRUE,
                                    do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_homogeneous_uniform_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  // FIXME: Should we maintain this contition?
  CAPI_EXPECTS((!options_cpp.retain_seeds_) || (starting_vertex_label_offsets != nullptr),
               CUGRAPH_INVALID_INPUT,
               "must specify starting_vertex_label_offsets if retain_seeds is true",
               *error);

  CAPI_EXPECTS((starting_vertex_label_offsets == nullptr) ||
                 (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                    starting_vertex_label_offsets)
                    ->type_ == SIZE_T),
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_label_offsets should be of type size_t",
               *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out type must be INT32",
    *error);

  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  neighbor_sampling_functor functor{handle,
                                    rng_state,
                                    graph,
                                    nullptr,
                                    start_vertices,
                                    starting_vertex_label_offsets,
                                    nullptr,
                                    fan_out,
                                    1,  // num_edge_types
                                    std::move(options_cpp),
                                    FALSE,
                                    do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_homogeneous_biased_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  CAPI_EXPECTS(
    (edge_biases != nullptr) ||
      (reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->edge_weights_ != nullptr),
    CUGRAPH_INVALID_INPUT,
    "edge_biases is required if the graph is not weighted",
    *error);

  // FIXME: Should we maintain this contition?
  CAPI_EXPECTS((!options_cpp.retain_seeds_) || (starting_vertex_label_offsets != nullptr),
               CUGRAPH_INVALID_INPUT,
               "must specify starting_vertex_label_offsets if retain_seeds is true",
               *error);

  CAPI_EXPECTS((starting_vertex_label_offsets == nullptr) ||
                 (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                    starting_vertex_label_offsets)
                    ->type_ == SIZE_T),
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_label_offsets should be of type size_t",
               *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out type must be INT32",
    *error);

  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  neighbor_sampling_functor functor{handle,
                                    rng_state,
                                    graph,
                                    edge_biases,
                                    start_vertices,
                                    starting_vertex_label_offsets,
                                    nullptr,
                                    fan_out,
                                    1,  // num_edge_types
                                    std::move(options_cpp),
                                    TRUE,
                                    do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

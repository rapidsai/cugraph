/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/properties.hpp"
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/sampling_common.hpp"
#include "c_api/utils.hpp"
#include "sampling/detail/sampling_utils.hpp"
#include "sampling/windowed_temporal_sampling_impl.hpp"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/sampling_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <raft/core/handle.hpp>

namespace {

struct temporal_neighbor_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  std::string temporal_column_name_{};
  cugraph::c_api::cugraph_edge_property_view_t const* edge_biases_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* starting_vertex_times_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* starting_vertex_label_offsets_{
    nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertex_type_offsets_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  int num_edge_types_{};
  cugraph::c_api::cugraph_sampling_options_t options_{};
  bool is_biased_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  // Window-based filtering parameters (B+C+D optimizations)
  bool use_windowed_sampling_{false};
  int64_t window_start_{0};
  int64_t window_end_{0};

  temporal_neighbor_sampling_functor(
    cugraph_resource_handle_t const* handle,
    cugraph_rng_state_t* rng_state,
    cugraph_graph_t* graph,
    const char* temporal_column_name,
    cugraph_edge_property_view_t const* edge_biases,
    cugraph_type_erased_device_array_view_t const* start_vertices,
    cugraph_type_erased_device_array_view_t const* starting_vertex_times,
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
      starting_vertex_times_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          starting_vertex_times)),
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
      do_expensive_check_(do_expensive_check),
      temporal_column_name_(temporal_column_name)
  {
  }

  void set_window_parameters(int64_t window_start, int64_t window_end)
  {
    use_windowed_sampling_ = true;
    window_start_          = window_start;
    window_end_            = window_end;
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

      auto edge_start_times = reinterpret_cast<cugraph::edge_property_t<edge_t, time_stamp_t>*>(
        graph_->edge_start_times_);

      if (edge_start_times == nullptr) {
        mark_error(CUGRAPH_INVALID_INPUT, "edge_start_times is required for temporal sampling");
        return;
      }

      auto edge_end_times =
        reinterpret_cast<cugraph::edge_property_t<edge_t, time_stamp_t>*>(graph_->edge_end_times_);

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

      std::optional<rmm::device_uvector<time_stamp_t>> starting_vertex_times{std::nullopt};
      if (starting_vertex_times_ != nullptr) {
        starting_vertex_times =
          rmm::device_uvector<time_stamp_t>(starting_vertex_times_->size_, handle_.get_stream());
        raft::copy(starting_vertex_times->data(),
                   starting_vertex_times_->as_type<time_stamp_t>(),
                   starting_vertex_times->size(),
                   handle_.get_stream());
      }

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

          std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
          if (starting_vertex_times) {
            vertex_properties.push_back(std::move(*starting_vertex_times));
          }
          if (start_vertex_labels) { vertex_properties.push_back(std::move(*start_vertex_labels)); }

          std::tie(start_vertices, vertex_properties) = cugraph::shuffle_ext_vertices(
            handle_, std::move(start_vertices), std::move(vertex_properties));

          size_t pos = 0;
          if (starting_vertex_times) {
            starting_vertex_times =
              std::move(std::get<rmm::device_uvector<time_stamp_t>>(vertex_properties[pos++]));
          }
          if (start_vertex_labels) {
            start_vertex_labels =
              std::move(std::get<rmm::device_uvector<label_t>>(vertex_properties[pos++]));
          }
        }
      } else {
        if constexpr (multi_gpu) {
          std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
          if (starting_vertex_times) {
            vertex_properties.push_back(std::move(*starting_vertex_times));
          }

          std::tie(start_vertices, std::ignore) = cugraph::shuffle_ext_vertices(
            handle_, std::move(start_vertices), std::move(vertex_properties));

          if (starting_vertex_times) {
            starting_vertex_times =
              std::move(std::get<rmm::device_uvector<time_stamp_t>>(vertex_properties[0]));
          }
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

      rmm::device_uvector<vertex_t> sampled_edge_srcs(0, handle_.get_stream());
      rmm::device_uvector<vertex_t> sampled_edge_dsts(0, handle_.get_stream());
      std::optional<rmm::device_uvector<weight_t>> sampled_weights{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> sampled_edge_ids{std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> sampled_edge_types{std::nullopt};
      std::optional<rmm::device_uvector<time_stamp_t>> sampled_edge_start_times{std::nullopt};
      std::optional<rmm::device_uvector<time_stamp_t>> sampled_edge_end_times{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> hop{std::nullopt};
      std::optional<rmm::device_uvector<label_t>> edge_label{std::nullopt};
      std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

      cugraph::temporal_sampling_comparison_t temporal_sampling_comparison{};
      switch (options_.temporal_sampling_comparison_) {
        case cugraph_temporal_sampling_comparison_t::STRICTLY_INCREASING:
          temporal_sampling_comparison =
            cugraph::temporal_sampling_comparison_t::STRICTLY_INCREASING;
          break;
        case cugraph_temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
          temporal_sampling_comparison =
            cugraph::temporal_sampling_comparison_t::MONOTONICALLY_INCREASING;
          break;
        case cugraph_temporal_sampling_comparison_t::STRICTLY_DECREASING:
          temporal_sampling_comparison =
            cugraph::temporal_sampling_comparison_t::STRICTLY_DECREASING;
          break;
        case cugraph_temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
          temporal_sampling_comparison =
            cugraph::temporal_sampling_comparison_t::MONOTONICALLY_DECREASING;
          break;
        default: CUGRAPH_FAIL("Invalid temporal sampling comparison type");
      };

      // FIXME: For biased sampling, the user should pass either biases or edge weights,
      // otherwised throw an error and suggest the user to call uniform neighbor sample instead

      if (num_edge_types_ > 1) {
        CUGRAPH_EXPECTS(edge_types != nullptr,
                        "edge types are necessary for heterogeneous sampling.");

        // call heterogeneous temporal neighbor sample
        if (is_biased_) {
          std::tie(sampled_edge_srcs,
                   sampled_edge_dsts,
                   sampled_weights,
                   sampled_edge_ids,
                   sampled_edge_types,
                   sampled_edge_start_times,
                   sampled_edge_end_times,
                   hop,
                   offsets) =
            cugraph::heterogeneous_biased_temporal_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              edge_types->view(),
              edge_start_times->view(),
              (edge_end_times != nullptr) ? std::make_optional(edge_end_times->view())
                                          : std::nullopt,
              (edge_biases != nullptr) ? *edge_biases : edge_weights->view(),
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              starting_vertex_times
                ? std::make_optional<raft::device_span<time_stamp_t const>>(
                    starting_vertex_times->data(), starting_vertex_times->size())
                : std::nullopt,
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
                                        options_.return_hops_ == TRUE,
                                        options_.dedupe_sources_ == TRUE,
                                        options_.with_replacement_ == TRUE,
                                        temporal_sampling_comparison,
                                        options_.disjoint_sampling_ == TRUE},
              do_expensive_check_);
        } else {
          std::tie(sampled_edge_srcs,
                   sampled_edge_dsts,
                   sampled_weights,
                   sampled_edge_ids,
                   sampled_edge_types,
                   sampled_edge_start_times,
                   sampled_edge_end_times,
                   hop,
                   offsets) =
            cugraph::heterogeneous_uniform_temporal_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              edge_types->view(),
              edge_start_times->view(),
              (edge_end_times != nullptr) ? std::make_optional(edge_end_times->view())
                                          : std::nullopt,
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              starting_vertex_times
                ? std::make_optional<raft::device_span<time_stamp_t const>>(
                    starting_vertex_times->data(), starting_vertex_times->size())
                : std::nullopt,
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
                                        options_.return_hops_ == TRUE,
                                        options_.dedupe_sources_ == TRUE,
                                        options_.with_replacement_ == TRUE,
                                        temporal_sampling_comparison,
                                        options_.disjoint_sampling_ == TRUE},
              do_expensive_check_);
        }
      } else {
        // Call homogeneous temporal neighbor sample
        if (is_biased_) {
          std::tie(sampled_edge_srcs,
                   sampled_edge_dsts,
                   sampled_weights,
                   sampled_edge_ids,
                   sampled_edge_types,
                   sampled_edge_start_times,
                   sampled_edge_end_times,
                   hop,
                   offsets) =
            cugraph::homogeneous_biased_temporal_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
              edge_start_times->view(),
              (edge_end_times != nullptr) ? std::make_optional(edge_end_times->view())
                                          : std::nullopt,
              (edge_biases != nullptr) ? *edge_biases : edge_weights->view(),
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              starting_vertex_times
                ? std::make_optional<raft::device_span<time_stamp_t const>>(
                    starting_vertex_times->data(), starting_vertex_times->size())
                : std::nullopt,
              (starting_vertex_label_offsets_ != nullptr)
                ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                   (*start_vertex_labels).size())
                : std::nullopt,
              label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                     (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                 : std::nullopt,
              raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
              cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                        options_.return_hops_ == TRUE,
                                        options_.dedupe_sources_ == TRUE,
                                        options_.with_replacement_ == TRUE,
                                        temporal_sampling_comparison,
                                        options_.disjoint_sampling_ == TRUE},
              do_expensive_check_);
        } else if (use_windowed_sampling_) {
          // Windowed temporal sampling with B+C+D optimizations
          // B: O(log E) binary search for window bounds
          // C: O(ΔE) incremental updates (when window_state provided)
          // D: Inline temporal filtering during sampling
          //
          // Note: B+C+D path only instantiated for int64/int64 types due to thrust
          // template compatibility. Other types fall back to standard path.
          if constexpr (std::is_same_v<vertex_t, int64_t> && std::is_same_v<edge_t, int64_t>) {
            // Get or create cached window_state from graph object for O(ΔE) incremental updates
            using window_state_type = cugraph::detail::window_state_t<edge_t, time_stamp_t>;

            if (graph_->window_state_ == nullptr) {
              // First windowed call: allocate window_state (will be initialized in impl)
              graph_->window_state_ = new window_state_type(handle_.get_stream());
            }

            auto* cached_window_state = reinterpret_cast<window_state_type*>(graph_->window_state_);

            std::tie(sampled_edge_srcs,
                     sampled_edge_dsts,
                     sampled_weights,
                     sampled_edge_ids,
                     sampled_edge_types,
                     sampled_edge_start_times,
                     sampled_edge_end_times,
                     hop,
                     offsets) =
              cugraph::detail::windowed_temporal_neighbor_sample_impl<vertex_t,
                                                                      edge_t,
                                                                      weight_t,
                                                                      edge_type_t,
                                                                      time_stamp_t,
                                                                      weight_t,
                                                                      label_t,
                                                                      false,
                                                                      multi_gpu>(
                handle_,
                rng_state_->rng_state_,
                graph_view,
                (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
                (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
                (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
                edge_start_times->view(),
                (edge_end_times != nullptr) ? std::make_optional(edge_end_times->view())
                                            : std::nullopt,
                std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{
                  std::nullopt},  // edge_bias
                raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
                starting_vertex_times
                  ? std::make_optional<raft::device_span<time_stamp_t const>>(
                      starting_vertex_times->data(), starting_vertex_times->size())
                  : std::nullopt,
                (starting_vertex_label_offsets_ != nullptr)
                  ? std::make_optional<raft::device_span<label_t const>>(
                      (*start_vertex_labels).data(), (*start_vertex_labels).size())
                  : std::nullopt,
                label_to_comm_rank ? std::make_optional(raft::device_span<int32_t const>{
                                       (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                   : std::nullopt,
                raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
                std::make_optional(edge_type_t{1}),  // num_edge_types
                cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                          options_.return_hops_ == TRUE,
                                          options_.dedupe_sources_ == TRUE,
                                          options_.with_replacement_ == TRUE,
                                          temporal_sampling_comparison,
                                          options_.disjoint_sampling_ == TRUE},
                std::make_optional(static_cast<time_stamp_t>(window_start_)),
                std::make_optional(static_cast<time_stamp_t>(window_end_)),
                std::make_optional(std::ref(*cached_window_state)),
                do_expensive_check_);
          } else {
            // Fallback for non-int64 types: use standard temporal sampling
            // (window parameters are ignored - user should use int64 graph for B+C+D)
            std::tie(sampled_edge_srcs,
                     sampled_edge_dsts,
                     sampled_weights,
                     sampled_edge_ids,
                     sampled_edge_types,
                     sampled_edge_start_times,
                     sampled_edge_end_times,
                     hop,
                     offsets) =
              cugraph::homogeneous_uniform_temporal_neighbor_sample(
                handle_,
                rng_state_->rng_state_,
                graph_view,
                (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
                (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
                (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
                edge_start_times->view(),
                (edge_end_times != nullptr) ? std::make_optional(edge_end_times->view())
                                            : std::nullopt,
                raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
                starting_vertex_times
                  ? std::make_optional<raft::device_span<time_stamp_t const>>(
                      starting_vertex_times->data(), starting_vertex_times->size())
                  : std::nullopt,
                (starting_vertex_label_offsets_ != nullptr)
                  ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                     (*start_vertex_labels).size())
                  : std::nullopt,
                label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                       (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                   : std::nullopt,
                raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
                cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                          options_.return_hops_ == TRUE,
                                          options_.dedupe_sources_ == TRUE,
                                          options_.with_replacement_ == TRUE,
                                          temporal_sampling_comparison,
                                          options_.disjoint_sampling_ == TRUE},
                do_expensive_check_);
          }
        } else {
          std::tie(sampled_edge_srcs,
                   sampled_edge_dsts,
                   sampled_weights,
                   sampled_edge_ids,
                   sampled_edge_types,
                   sampled_edge_start_times,
                   sampled_edge_end_times,
                   hop,
                   offsets) =
            cugraph::homogeneous_uniform_temporal_neighbor_sample(
              handle_,
              rng_state_->rng_state_,
              graph_view,
              (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
              (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
              (edge_types != nullptr) ? std::make_optional(edge_types->view()) : std::nullopt,
              edge_start_times->view(),
              (edge_end_times != nullptr) ? std::make_optional(edge_end_times->view())
                                          : std::nullopt,
              raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
              starting_vertex_times
                ? std::make_optional<raft::device_span<time_stamp_t const>>(
                    starting_vertex_times->data(), starting_vertex_times->size())
                : std::nullopt,
              (starting_vertex_label_offsets_ != nullptr)
                ? std::make_optional<raft::device_span<int const>>((*start_vertex_labels).data(),
                                                                   (*start_vertex_labels).size())
                : std::nullopt,
              label_to_comm_rank ? std::make_optional(raft::device_span<int const>{
                                     (*label_to_comm_rank).data(), (*label_to_comm_rank).size()})
                                 : std::nullopt,
              raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
              cugraph::sampling_flags_t{options_.prior_sources_behavior_,
                                        options_.return_hops_ == TRUE,
                                        options_.dedupe_sources_ == TRUE,
                                        options_.with_replacement_ == TRUE,
                                        temporal_sampling_comparison,
                                        options_.disjoint_sampling_ == TRUE},
              do_expensive_check_);
        }
      }

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        sampled_edge_srcs.data(),
        sampled_edge_srcs.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        sampled_edge_dsts.data(),
        sampled_edge_dsts.size(),
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
        if (sampled_edge_srcs.size() > 0) {  // Only renumber if there are edgelist to renumber
          if (num_edge_types_ == 1) {        // homogeneous renumbering
            if (options_.compression_type_ == cugraph_compression_type_t::COO) {
              // COO

              rmm::device_uvector<vertex_t> output_majors(0, handle_.get_stream());
              rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());
              std::vector<cugraph::arithmetic_device_uvector_t> output_edge_properties{};
              if (sampled_weights) {
                output_edge_properties.push_back(std::move(*sampled_weights));
              }
              if (sampled_edge_ids) {
                output_edge_properties.push_back(std::move(*sampled_edge_ids));
              }
              if (sampled_edge_types) {
                output_edge_properties.push_back(std::move(*sampled_edge_types));
              }
              if (sampled_edge_start_times) {
                output_edge_properties.push_back(std::move(*sampled_edge_start_times));
              }
              if (sampled_edge_end_times) {
                output_edge_properties.push_back(std::move(*sampled_edge_end_times));
              }

              std::tie(output_majors,
                       minors,
                       output_edge_properties,
                       label_hop_offsets,
                       output_renumber_map,
                       renumber_map_offsets) =
                cugraph::renumber_and_sort_sampled_edgelist<vertex_t>(
                  handle_,
                  std::move(sampled_edge_srcs),
                  std::move(sampled_edge_dsts),
                  std::move(output_edge_properties),
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

              size_t pos = 0;
              if (sampled_weights) {
                sampled_weights =
                  std::move(std::get<rmm::device_uvector<weight_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_ids) {
                sampled_edge_ids =
                  std::move(std::get<rmm::device_uvector<edge_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_types) {
                sampled_edge_types = std::move(
                  std::get<rmm::device_uvector<edge_type_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_start_times) {
                sampled_edge_start_times = std::move(
                  std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_end_times) {
                sampled_edge_end_times = std::move(
                  std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
              }
            } else {
              // (D)CSC, (D)CSR

              bool doubly_compress =
                (options_.compression_type_ == cugraph_compression_type_t::DCSR) ||
                (options_.compression_type_ == cugraph_compression_type_t::DCSC);

              rmm::device_uvector<size_t> output_major_offsets(0, handle_.get_stream());
              rmm::device_uvector<vertex_t> output_renumber_map(0, handle_.get_stream());

              std::vector<cugraph::arithmetic_device_uvector_t> output_edge_properties{};
              if (sampled_weights) {
                output_edge_properties.push_back(std::move(*sampled_weights));
              }
              if (sampled_edge_ids) {
                output_edge_properties.push_back(std::move(*sampled_edge_ids));
              }
              if (sampled_edge_start_times) {
                output_edge_properties.push_back(std::move(*sampled_edge_start_times));
              }
              if (sampled_edge_end_times) {
                output_edge_properties.push_back(std::move(*sampled_edge_end_times));
              }
              std::tie(majors,
                       output_major_offsets,
                       minors,
                       output_edge_properties,
                       sampled_edge_types,
                       label_hop_offsets,
                       output_renumber_map,
                       renumber_map_offsets) =
                cugraph::renumber_and_compress_sampled_edgelist(
                  handle_,
                  std::move(sampled_edge_srcs),
                  std::move(sampled_edge_dsts),
                  std::move(output_edge_properties),
                  std::move(sampled_edge_types),
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

              size_t pos = 0;
              if (sampled_weights) {
                sampled_weights =
                  std::move(std::get<rmm::device_uvector<weight_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_ids) {
                sampled_edge_ids =
                  std::move(std::get<rmm::device_uvector<edge_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_start_times) {
                sampled_edge_start_times = std::move(
                  std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
              }
              if (sampled_edge_end_times) {
                sampled_edge_end_times = std::move(
                  std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
              }

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

            std::vector<cugraph::arithmetic_device_uvector_t> output_edge_properties{};
            if (sampled_weights) { output_edge_properties.push_back(std::move(*sampled_weights)); }
            if (sampled_edge_start_times) {
              output_edge_properties.push_back(std::move(*sampled_edge_start_times));
            }
            if (sampled_edge_end_times) {
              output_edge_properties.push_back(std::move(*sampled_edge_end_times));
            }

            std::tie(output_majors,
                     minors,
                     output_edge_properties,
                     sampled_edge_ids,
                     label_type_hop_offsets,  // Contains information about the type and hop offsets
                     output_renumber_map,
                     renumber_map_offsets,
                     renumbered_and_sorted_edge_id_renumber_map,
                     renumbered_and_sorted_edge_id_renumber_map_label_type_offsets) =
              cugraph::heterogeneous_renumber_and_sort_sampled_edgelist<vertex_t>(
                handle_,
                std::move(sampled_edge_srcs),
                std::move(sampled_edge_dsts),
                std::move(output_edge_properties),
                std::move(sampled_edge_ids),
                std::move(sampled_edge_types),
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
            if (sampled_edge_types) {
              (*sampled_edge_types)
                .resize(raft::device_span<size_t const>{(*label_type_hop_offsets).data(),
                                                        (*label_type_hop_offsets).size()}
                            .back() -
                          1,
                        handle_.get_stream());
              cugraph::detail::sequence_fill(handle_.get_stream(),
                                             (*sampled_edge_types).begin(),
                                             (*sampled_edge_types).size(),
                                             edge_type_t{0});
            }

            size_t pos = 0;
            if (sampled_weights) {
              sampled_weights =
                std::move(std::get<rmm::device_uvector<weight_t>>(output_edge_properties[pos++]));
            }
            if (sampled_edge_start_times) {
              sampled_edge_start_times = std::move(
                std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
            }
            if (sampled_edge_end_times) {
              sampled_edge_end_times = std::move(
                std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
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

        std::vector<cugraph::arithmetic_device_uvector_t> output_edge_properties{};
        if (sampled_weights) { output_edge_properties.push_back(std::move(*sampled_weights)); }
        if (sampled_edge_ids) { output_edge_properties.push_back(std::move(*sampled_edge_ids)); }
        if (sampled_edge_types) {
          output_edge_properties.push_back(std::move(*sampled_edge_types));
        }
        if (sampled_edge_start_times) {
          output_edge_properties.push_back(std::move(*sampled_edge_start_times));
        }
        if (sampled_edge_end_times) {
          output_edge_properties.push_back(std::move(*sampled_edge_end_times));
        }

        std::tie(sampled_edge_srcs, sampled_edge_dsts, output_edge_properties, label_hop_offsets) =
          cugraph::sort_sampled_edgelist(handle_,
                                         std::move(sampled_edge_srcs),
                                         std::move(sampled_edge_dsts),
                                         std::move(output_edge_properties),
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

        majors.emplace(std::move(sampled_edge_srcs));
        minors = std::move(sampled_edge_dsts);

        size_t pos = 0;
        if (sampled_weights) {
          sampled_weights =
            std::move(std::get<rmm::device_uvector<weight_t>>(output_edge_properties[pos++]));
        }
        if (sampled_edge_ids) {
          sampled_edge_ids =
            std::move(std::get<rmm::device_uvector<edge_t>>(output_edge_properties[pos++]));
        }
        if (sampled_edge_types) {
          sampled_edge_types =
            std::move(std::get<rmm::device_uvector<edge_type_t>>(output_edge_properties[pos++]));
        }
        if (sampled_edge_start_times) {
          sampled_edge_start_times =
            std::move(std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
        }
        if (sampled_edge_end_times) {
          sampled_edge_end_times =
            std::move(std::get<rmm::device_uvector<time_stamp_t>>(output_edge_properties[pos++]));
        }

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
        (sampled_edge_ids) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                               *sampled_edge_ids, graph_->edge_type_)
                           : nullptr,
        (sampled_edge_types) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                 *sampled_edge_types, graph_->edge_type_id_type_)
                             : nullptr,
        (sampled_weights) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                              *sampled_weights, graph_->weight_type_)
                          : nullptr,
        (sampled_edge_start_times) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                       *sampled_edge_start_times, graph_->edge_time_type_)
                                   : nullptr,
        (sampled_edge_end_times) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                                     *sampled_edge_end_times, graph_->edge_time_type_)
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

cugraph_error_code_t cugraph_heterogeneous_uniform_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* temporal_column_name,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_times,
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

  // FIXME: Should we maintain this condition?
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

  CAPI_EXPECTS(starting_vertex_times == nullptr ||
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   starting_vertex_times)
                     ->size_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     start_vertices)
                     ->size_,
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_times should have the same size as start_vertices",
               *error);

  temporal_neighbor_sampling_functor functor{handle,
                                             rng_state,
                                             graph,
                                             temporal_column_name,
                                             nullptr,
                                             start_vertices,
                                             starting_vertex_times,
                                             starting_vertex_label_offsets,
                                             vertex_type_offsets,
                                             fan_out,
                                             num_edge_types,
                                             std::move(options_cpp),
                                             FALSE,
                                             do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_heterogeneous_biased_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* temporal_column_name,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_times,
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

  // FIXME: Should we maintain this condition?
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

  CAPI_EXPECTS(starting_vertex_times == nullptr ||
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   starting_vertex_times)
                     ->size_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     start_vertices)
                     ->size_,
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_times should have the same size as start_vertices",
               *error);

  temporal_neighbor_sampling_functor functor{handle,
                                             rng_state,
                                             graph,
                                             temporal_column_name,
                                             edge_biases,
                                             start_vertices,
                                             starting_vertex_times,
                                             starting_vertex_label_offsets,
                                             vertex_type_offsets,
                                             fan_out,
                                             num_edge_types,
                                             std::move(options_cpp),
                                             TRUE,
                                             do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_homogeneous_uniform_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* temporal_column_name,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_times,
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

  CAPI_EXPECTS(starting_vertex_times == nullptr ||
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   starting_vertex_times)
                     ->size_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     start_vertices)
                     ->size_,
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_times should have the same size as start_vertices",
               *error);

  temporal_neighbor_sampling_functor functor{handle,
                                             rng_state,
                                             graph,
                                             temporal_column_name,
                                             nullptr,
                                             start_vertices,
                                             starting_vertex_times,
                                             starting_vertex_label_offsets,
                                             nullptr,
                                             fan_out,
                                             1,  // num_edge_types
                                             std::move(options_cpp),
                                             FALSE,
                                             do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_homogeneous_biased_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* temporal_column_name,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_times,
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

  CAPI_EXPECTS(starting_vertex_times == nullptr ||
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   starting_vertex_times)
                     ->size_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     start_vertices)
                     ->size_,
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_times should have the same size as start_vertices",
               *error);

  temporal_neighbor_sampling_functor functor{handle,
                                             rng_state,
                                             graph,
                                             temporal_column_name,
                                             edge_biases,
                                             start_vertices,
                                             starting_vertex_times,
                                             starting_vertex_label_offsets,
                                             nullptr,
                                             fan_out,
                                             1,  // num_edge_types
                                             std::move(options_cpp),
                                             TRUE,
                                             do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_homogeneous_uniform_temporal_neighbor_sample_windowed(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* temporal_column_name,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  const cugraph_sampling_options_t* options,
  int64_t window_start,
  int64_t window_end,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  auto options_cpp = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);

  // Validate window parameters
  CAPI_EXPECTS(window_end > window_start,
               CUGRAPH_INVALID_INPUT,
               "window_end must be greater than window_start",
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

  CAPI_EXPECTS(starting_vertex_times == nullptr ||
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   starting_vertex_times)
                     ->size_ ==
                   reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                     start_vertices)
                     ->size_,
               CUGRAPH_INVALID_INPUT,
               "starting_vertex_times should have the same size as start_vertices",
               *error);

  temporal_neighbor_sampling_functor functor{handle,
                                             rng_state,
                                             graph,
                                             temporal_column_name,
                                             nullptr,  // edge_biases
                                             start_vertices,
                                             starting_vertex_times,
                                             starting_vertex_label_offsets,
                                             nullptr,  // vertex_type_offsets
                                             fan_out,
                                             1,  // num_edge_types
                                             std::move(options_cpp),
                                             FALSE,  // is_biased
                                             do_expensive_check};

  // Enable windowed sampling with B+C+D optimizations
  functor.set_window_parameters(window_start, window_end);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

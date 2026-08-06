/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/sampling_algorithms.h>

#include <cugraph/sampling_functions.hpp>

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
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison_{
    cugraph_temporal_sampling_comparison_t::STRICTLY_INCREASING};
  bool_t disjoint_sampling_{FALSE};
  cugraph::neighbor_selection_t neighbor_selection_{cugraph::neighbor_selection_t::RANDOM};
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
  cugraph_type_erased_device_array_t* edge_start_time_{nullptr};
  cugraph_type_erased_device_array_t* edge_end_time_{nullptr};
  cugraph_type_erased_device_array_t* hop_{nullptr};
  cugraph_type_erased_device_array_t* label_hop_offsets_{nullptr};
  cugraph_type_erased_device_array_t* label_type_hop_offsets_{nullptr};
  cugraph_type_erased_device_array_t* label_{nullptr};
  cugraph_type_erased_device_array_t* renumber_map_{nullptr};
  cugraph_type_erased_device_array_t* renumber_map_offsets_{nullptr};
  cugraph_type_erased_device_array_t* edge_renumber_map_{nullptr};
  cugraph_type_erased_device_array_t* edge_renumber_map_offsets_{nullptr};
};

inline cugraph_neighbor_selection_t to_c_neighbor_selection(
  cugraph::neighbor_selection_t neighbor_selection)
{
  switch (neighbor_selection) {
    case cugraph::neighbor_selection_t::FIRST: return CUGRAPH_NEIGHBOR_SELECTION_FIRST;
    case cugraph::neighbor_selection_t::LAST: return CUGRAPH_NEIGHBOR_SELECTION_LAST;
    default: return CUGRAPH_NEIGHBOR_SELECTION_RANDOM;
  }
}

// Functors live in anonymous namespaces in separate .cpp files; these helpers let
// cugraph_neighbor_sample invoke either path without calling the deprecated C entry points.
// Use ::-qualified C opaque types so they are not shadowed by cugraph::c_api::* structs.
// Shared implementation for cugraph_neighbor_sample and biased legacy wrappers.
cugraph_error_code_t run_neighbor_sample(
  ::cugraph_resource_handle_t const* handle,
  ::cugraph_rng_state_t* rng_state,
  ::cugraph_graph_t* graph,
  ::cugraph_edge_property_view_t const* edge_biases,
  ::cugraph_type_erased_device_array_view_t const* start_vertices,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_start_times,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_end_times,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_label_offsets,
  ::cugraph_type_erased_device_array_view_t const* vertex_type_offsets,
  ::cugraph_type_erased_host_array_view_t const* fan_out,
  int num_edge_types,
  cugraph_neighbor_selection_t neighbor_selection,
  cugraph_temporal_sampling_comparison_t const* temporal_sampling_comparison,
  ::cugraph_sampling_options_t const* sampling_options,
  bool is_biased,
  bool_t do_expensive_check,
  ::cugraph_sample_result_t** result,
  ::cugraph_error_t** error);

cugraph_error_code_t dispatch_non_temporal_neighbor_sample(
  ::cugraph_resource_handle_t const* handle,
  ::cugraph_rng_state_t* rng_state,
  ::cugraph_graph_t* graph,
  ::cugraph_edge_property_view_t const* edge_biases,
  ::cugraph_type_erased_device_array_view_t const* start_vertices,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_label_offsets,
  ::cugraph_type_erased_device_array_view_t const* vertex_type_offsets,
  ::cugraph_type_erased_host_array_view_t const* fan_out,
  int num_edge_types,
  cugraph_sampling_options_t options,
  bool is_biased,
  bool_t do_expensive_check,
  ::cugraph_sample_result_t** result,
  ::cugraph_error_t** error);

cugraph_error_code_t dispatch_temporal_neighbor_sample(
  ::cugraph_resource_handle_t const* handle,
  ::cugraph_rng_state_t* rng_state,
  ::cugraph_graph_t* graph,
  char const* temporal_column_name,
  ::cugraph_edge_property_view_t const* edge_biases,
  ::cugraph_type_erased_device_array_view_t const* start_vertices,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_start_times,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_end_times,
  ::cugraph_type_erased_device_array_view_t const* starting_vertex_label_offsets,
  ::cugraph_type_erased_device_array_view_t const* vertex_type_offsets,
  ::cugraph_type_erased_host_array_view_t const* fan_out,
  int num_edge_types,
  cugraph_sampling_options_t options,
  bool is_biased,
  bool_t do_expensive_check,
  ::cugraph_sample_result_t** result,
  ::cugraph_error_t** error);

}  // namespace c_api
}  // namespace cugraph

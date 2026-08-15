/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/sampling_common.hpp"

#include <cugraph_c/sampling_algorithms.h>

extern "C" cugraph_error_code_t cugraph_heterogeneous_uniform_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* /* temporal_column_name */,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_start_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_end_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_device_array_view_t* vertex_type_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  int num_edge_types,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  // Legacy temporal entry points imply temporal mode; set it on a copy so the caller's
  // options are left untouched.
  auto options_copy = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);
  options_copy.temporal_sampling_enabled_ = TRUE;
  return cugraph_neighbor_sample(handle,
                                 rng_state,
                                 graph,
                                 nullptr,
                                 start_vertices,
                                 starting_vertex_start_times,
                                 starting_vertex_end_times,
                                 starting_vertex_label_offsets,
                                 vertex_type_offsets,
                                 fan_out,
                                 num_edge_types,
                                 reinterpret_cast<cugraph_sampling_options_t const*>(&options_copy),
                                 do_expensive_check,
                                 result,
                                 error);
}

extern "C" cugraph_error_code_t cugraph_heterogeneous_biased_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* /* temporal_column_name */,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_start_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_end_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_device_array_view_t* vertex_type_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  int num_edge_types,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  // Legacy temporal entry points imply temporal mode; set it on a copy so the caller's
  // options are left untouched.
  auto options_copy = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);
  options_copy.temporal_sampling_enabled_ = TRUE;
  return cugraph_neighbor_sample(handle,
                                 rng_state,
                                 graph,
                                 edge_biases,
                                 start_vertices,
                                 starting_vertex_start_times,
                                 starting_vertex_end_times,
                                 starting_vertex_label_offsets,
                                 vertex_type_offsets,
                                 fan_out,
                                 num_edge_types,
                                 reinterpret_cast<cugraph_sampling_options_t const*>(&options_copy),
                                 do_expensive_check,
                                 result,
                                 error);
}

extern "C" cugraph_error_code_t cugraph_homogeneous_uniform_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* /* temporal_column_name */,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_start_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_end_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  // Legacy temporal entry points imply temporal mode; set it on a copy so the caller's
  // options are left untouched.
  auto options_copy = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);
  options_copy.temporal_sampling_enabled_ = TRUE;
  return cugraph_neighbor_sample(handle,
                                 rng_state,
                                 graph,
                                 nullptr,
                                 start_vertices,
                                 starting_vertex_start_times,
                                 starting_vertex_end_times,
                                 starting_vertex_label_offsets,
                                 nullptr,
                                 fan_out,
                                 1,
                                 reinterpret_cast<cugraph_sampling_options_t const*>(&options_copy),
                                 do_expensive_check,
                                 result,
                                 error);
}

extern "C" cugraph_error_code_t cugraph_homogeneous_biased_temporal_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const char* /* temporal_column_name */,
  const cugraph_edge_property_view_t* edge_biases,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  const cugraph_type_erased_device_array_view_t* starting_vertex_start_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_end_times,
  const cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets,
  const cugraph_type_erased_host_array_view_t* fan_out,
  const cugraph_sampling_options_t* options,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  // Legacy temporal entry points imply temporal mode; set it on a copy so the caller's
  // options are left untouched.
  auto options_copy = *reinterpret_cast<cugraph::c_api::cugraph_sampling_options_t const*>(options);
  options_copy.temporal_sampling_enabled_ = TRUE;
  return cugraph_neighbor_sample(handle,
                                 rng_state,
                                 graph,
                                 edge_biases,
                                 start_vertices,
                                 starting_vertex_start_times,
                                 starting_vertex_end_times,
                                 starting_vertex_label_offsets,
                                 nullptr,
                                 fan_out,
                                 1,
                                 reinterpret_cast<cugraph_sampling_options_t const*>(&options_copy),
                                 do_expensive_check,
                                 result,
                                 error);
}

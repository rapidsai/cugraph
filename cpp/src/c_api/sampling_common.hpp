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
  bool_t temporal_sampling_enabled_{FALSE};
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison_{
    cugraph_temporal_sampling_comparison_t::STRICTLY_INCREASING};
  bool_t disjoint_sampling_{FALSE};
  bool_t use_edge_weights_as_biases_{FALSE};
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
    case cugraph::neighbor_selection_t::LAST: return CUGRAPH_NEIGHBOR_SELECTION_LAST;
    default: return CUGRAPH_NEIGHBOR_SELECTION_RANDOM;
  }
}

}  // namespace c_api
}  // namespace cugraph

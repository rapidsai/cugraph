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

}  // namespace c_api
}  // namespace cugraph

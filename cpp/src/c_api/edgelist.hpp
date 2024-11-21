/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cugraph_c/edgelist.h>

#include <c_api/array.hpp>

#include <limits>

namespace cugraph {
namespace c_api {

struct cugraph_edgelist_t {
  size_t num_lists_{1};
  bool store_transposed_{false};
  size_t num_edges_{std::numeric_limits<size_t>::max()};

  std::unique_ptr<::cugraph_type_erased_device_array_t> offsets_{};
  std::unique_ptr<::cugraph_type_erased_device_array_t> indices_{};

  std::vector<::cugraph_type_erased_device_array_t*> vertices_{};
  std::vector<::cugraph_type_erased_device_array_t*> majors_{};
  std::vector<::cugraph_type_erased_device_array_t*> minors_{};
  std::vector<::cugraph_type_erased_device_array_t*> weights_{};
  std::vector<::cugraph_type_erased_device_array_t*> edge_ids_{};
  std::vector<::cugraph_type_erased_device_array_t*> edge_types_{};
  std::vector<::cugraph_type_erased_device_array_t*> edge_start_times_{};
  std::vector<::cugraph_type_erased_device_array_t*> edge_end_times_{};

  cugraph_data_type_id_t vertex_type_{NTYPES};
  cugraph_data_type_id_t edge_weight_type_{NTYPES};
  cugraph_data_type_id_t edge_id_type_{NTYPES};
  cugraph_data_type_id_t edge_type_type_{NTYPES};
  cugraph_data_type_id_t edge_time_type_{NTYPES};
};

}  // namespace c_api
}  // namespace cugraph

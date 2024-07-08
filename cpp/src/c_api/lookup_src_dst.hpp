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

#include "c_api/array.hpp"
#include "c_api/error.hpp"

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <memory>

namespace cugraph {
namespace c_api {

struct cugraph_lookup_container_t {
  cugraph_data_type_id_t edge_type_;
  cugraph_data_type_id_t edge_type_id_type_;
  cugraph_data_type_id_t vertex_type_;

  void* lookup_container_;
};

struct cugraph_lookup_result_t {
  cugraph_type_erased_device_array_t* srcs_{nullptr};
  cugraph_type_erased_device_array_t* dsts_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

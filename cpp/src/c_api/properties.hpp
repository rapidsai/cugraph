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

#include <cugraph_c/properties.h>

namespace cugraph {
namespace c_api {

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* vertex_property_;
} cugraph_vertex_property_t;

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* edge_property_;
} cugraph_edge_property_t;

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* vertex_property_;
} cugraph_vertex_property_view_t;

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* edge_property_;
} cugraph_edge_property_view_t;

}  // namespace c_api
}  // namespace cugraph

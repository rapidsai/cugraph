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

#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} lookup_container_t;

typedef struct {
  int32_t align_;
} lookup_result_t;

cugraph_error_code_t cugraph_build_edge_id_and_type_to_src_dst_lookup_map(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  lookup_container_t** lookup_container,
  cugraph_error_t** error);

cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_single_type(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  int edge_type_to_lookup,
  lookup_result_t** result,
  cugraph_error_t** error);

cugraph_error_code_t cugraph_lookup_endpoints_from_edge_ids_and_types(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const lookup_container_t* lookup_container,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  const cugraph_type_erased_device_array_view_t* edge_types_to_lookup,
  lookup_result_t** result,
  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif

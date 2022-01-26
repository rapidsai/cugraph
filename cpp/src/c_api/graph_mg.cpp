/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph_c/graph.h>

// FIXME: assume that this function will directly call
// create_graph_from_edgelist() instead of invoking a graph constructor, in
// which case, some parameters here (e.g. vertex_partition_offsets,
// segment_offsets) are related to implementation details and unnecessary if
// this function calls create_graph_from_edgelist().

extern "C" cugraph_error_code_t cugraph_mg_graph_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  const cugraph_type_erased_host_array_view_t* vertex_partition_offsets,
  const cugraph_type_erased_host_array_view_t* segment_offsets,
  bool_t store_transposed,
  size_t num_vertices,
  size_t num_edges,
  bool_t check,
  cugraph_graph_t** graph,
  cugraph_error_t** error)
{
  *graph = nullptr;
  return CUGRAPH_NOT_IMPLEMENTED;
}

extern "C" void cugraph_mg_graph_free(cugraph_graph_t* ptr_graph) {}

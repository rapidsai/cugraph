/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

extern "C" cugraph_error_code_t cugraph_mg_graph_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_properties_t* properties,
  const cugraph_type_erased_device_array_t* src,
  const cugraph_type_erased_device_array_t* dst,
  const cugraph_type_erased_device_array_t* weights,
  const cugraph_type_erased_host_array_t* vertex_partition_offsets,
  const cugraph_type_erased_host_array_t* segment_offsets,
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

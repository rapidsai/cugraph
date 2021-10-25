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

extern "C" cugraph_graph_t* cugraph_sg_graph_create(
  const cugraph_raft_handle_t* p_handle,
  const cugraph_type_erased_device_array_t* src,
  const cugraph_type_erased_device_array_t* dst,
  const cugraph_type_erased_device_array_t* weights,
  bool_t store_transposed,
  size_t num_vertices,
  size_t num_edges,
  bool_t check,
  bool_t is_symmetric,
  bool_t is_multigraph)
{
  return nullptr;
}

extern "C" void cugraph_sg_graph_free(cugraph_graph_t* ptr_graph) {}

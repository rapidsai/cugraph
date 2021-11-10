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

#include <cugraph_c/algorithms.h>

namespace c_api {
struct cugraph_bfs_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_;
  cugraph_type_erased_device_array_t* distances_;
  cugraph_type_erased_device_array_t* predecessors_;
};

}  // namespace c_api

extern "C" cugraph_type_erased_device_array_t* cugraph_bfs_result_get_vertices(
  cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_bfs_result_t*>(result);
  return internal_pointer->vertex_ids_;
}

extern "C" cugraph_type_erased_device_array_t* cugraph_bfs_result_get_distances(
  cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_bfs_result_t*>(result);
  return internal_pointer->distances_;
}

extern "C" cugraph_type_erased_device_array_t* cugraph_bfs_result_get_predecessors(
  cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_bfs_result_t*>(result);
  return internal_pointer->predecessors_;
}

extern "C" void cugraph_bfs_result_free(cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_bfs_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->distances_;
  delete internal_pointer->predecessors_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_bfs(const cugraph_resource_handle_t* handle,
                                            const cugraph_graph_t* graph,
                                            const cugraph_type_erased_device_array_t* sources,
                                            bool_t direction_optimizing,
                                            size_t depth_limit,
                                            bool_t do_expensive_check,
                                            bool_t compute_predecessors,
                                            cugraph_bfs_result_t** result,
                                            cugraph_error_t** error)
{
  return CUGRAPH_NOT_IMPLEMENTED;
}

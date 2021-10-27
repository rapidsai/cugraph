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

cugraph_error_t bfs(const cugraph_handle_t* handle,
                    const cugraph_graph_t* graph,
                    const cugraph_type_erased_device_array_t* sources,
                    bool direction_optimizing,
                    size_t depth_limit,
                    bool do_expensive_check,
                    cugraph_type_erased_device_array_t** vertex_ids,
                    cugraph_type_erased_device_array_t** distances,
                    cugraph_type_erased_device_array_t** predecessors);
{
  return CUGRAPH_NOT_IMPLEMENTED;
}

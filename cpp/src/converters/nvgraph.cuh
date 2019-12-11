/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <nvgraph/nvgraph.h>
#include <cugraph.h>

namespace cugraph {
/**
 * Takes a GDF graph and wraps its data with an Nvgraph graph object.
 * @param nvg_handle The Nvgraph handle
 * @param gdf_G Pointer to GDF graph object
 * @param nvgraph_G Pointer to the Nvgraph graph descriptor
 * @param use_transposed True if we are transposing the input graph while wrapping
 * @return Error code
 */
void createGraph_nvgraph(nvgraphHandle_t nvg_handle,
                                  Graph* gdf_G,
                                  nvgraphGraphDescr_t * nvgraph_G,
bool use_transposed = false);
}
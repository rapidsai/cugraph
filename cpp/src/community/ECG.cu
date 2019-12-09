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
/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_gdf.cu
 * ---------------------------------------------------------------------------**/

#include <cugraph.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include <ctime>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"
#include <rmm_utils.h>

namespace cugraph {
void ecg(Graph* graph,
         double min_weight,
         int ensemble_size,
         gdf_column *ecg_parts) {

}
}

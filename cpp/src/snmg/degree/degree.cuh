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

#pragma once
#include <omp.h>
#include "utilities/graph_utils.cuh"
#include "snmg/utils.cuh"
#include "rmm_utils.h"

namespace cugraph {
  /**
   * Single node multi-GPU method for degree calculation on a partitioned graph.
   * @param x Indicates whether to compute in degree, out degree, or the sum of both.
   *    0 = in + out degree
   *    1 = in-degree
   *    2 = out-degree
   * @param part_off The vertex partitioning of the global graph
   * @param off The offsets array of the local partition
   * @param ind The indices array of the local partition
   * @param degree Pointer to pointers to memory on each GPU for the result
   * @return Error code
   */
  template<typename idx_t>
  gdf_error snmg_degree(int x, size_t* part_off, idx_t* off, idx_t* ind, idx_t** degree);

}

/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// Author: Xavier Cadet xcadet@nvidia.com
#include <cugraph.h>
#include <rmm_utils.h>

#include "betweenness_centrality.cuh"

namespace cugraph {
namespace detail {

template <typename IndexType, typename DistType>
void BC<IndexType, DistType>::setup() {
}

} //namespace detail
  /**
  * ---------------------------------------------------------------------------*
  * @brief Native betweenness centrality
  *
  * @file betweenness_centrality.cu
  * --------------------------------------------------------------------------*/
  //template void betweenness_centrality<int, float>();
  //template void betweenness_centrality<int, float>();
} //namespace cugraph
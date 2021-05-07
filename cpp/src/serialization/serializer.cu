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

// Andrei Schaffer, aschaffer@nvidia.com
//

#include <serialization/serializer.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/copy.h>

namespace cugraph {
namespace serializer {
}  // namespace serializer
}  // namespace cugraph


/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuco/hash_functions.cuh>

#include <cuda/std/tuple>

template <typename vertex_t>
struct hash_vertex_pair_t {
  using result_type = typename cuco::murmurhash3_32<vertex_t>::result_type;

  __device__ result_type operator()(cuda::std::tuple<vertex_t, vertex_t> const& pair) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
    auto hash0 = hash_func(thrust::get<0>(pair));
    auto hash1 = hash_func(thrust::get<1>(pair));
    return hash0 + hash1;
  }
};


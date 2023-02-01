/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

namespace cugraph {
namespace detail {

template __device__ float parallel_prefix_sum(int32_t, int32_t const*, float const*);
template __device__ double parallel_prefix_sum(int32_t, int32_t const*, double const*);
template __device__ float parallel_prefix_sum(int64_t, int32_t const*, float const*);
template __device__ double parallel_prefix_sum(int64_t, int32_t const*, double const*);
template __device__ float parallel_prefix_sum(int64_t, int64_t const*, float const*);
template __device__ double parallel_prefix_sum(int64_t, int64_t const*, double const*);

template void offsets_to_indices<int32_t, int32_t>(int32_t const*, int32_t, int32_t*);
template void offsets_to_indices<int64_t, int32_t>(int64_t const*, int32_t, int32_t*);
template void offsets_to_indices<int64_t, int64_t>(int64_t const*, int64_t, int64_t*);

template __global__ void offsets_to_indices_kernel<int32_t, int32_t>(int32_t const*,
                                                                     int32_t,
                                                                     int32_t*);
template __global__ void offsets_to_indices_kernel<int64_t, int32_t>(int64_t const*,
                                                                     int32_t,
                                                                     int32_t*);
template __global__ void offsets_to_indices_kernel<int64_t, int64_t>(int64_t const*,
                                                                     int64_t,
                                                                     int64_t*);

}  // namespace detail
}  // namespace cugraph

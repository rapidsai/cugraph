/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <tuple>
#include <vector>

namespace cugraph {

namespace detail {

template <typename input_t, typename output_t>
struct typecast_t {
  __device__ output_t operator()(input_t val) const { return static_cast<output_t>(val); }
};

template <typename T>
struct not_equal_t {
  T compare{};

  __device__ bool operator()(T val) const { return val != compare; }
};

template <typename T>
struct multiplier_t {
  T multiplier{};

  __device__ T operator()(T input) const { return input * multiplier; }
};

}  // namespace detail

}  // namespace cugraph

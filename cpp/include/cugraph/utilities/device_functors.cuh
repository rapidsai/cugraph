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

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstddef>
#include <tuple>
#include <vector>

namespace cugraph {

namespace detail {

template <typename input_t, typename output_t>
struct typecast_t {
  __device__ output_t operator()(input_t val) const { return static_cast<output_t>(val); }
};

template <typename Iterator>
struct indirection_t {
  Iterator first{};

  __device__ typename thrust::iterator_traits<Iterator>::value_type operator()(size_t i) const
  {
    return *(first + i);
  }
};

template <typename T>
struct not_equal_t {
  T compare{};

  __device__ bool operator()(T val) const { return val != compare; }
};

template <typename Iterator>
struct is_first_in_run_t {
  Iterator iter{};

  __device__ bool operator()(size_t i) const
  {
    return (i == 0) || (*(iter + (i - 1)) != *(iter + i));
  }
};

template <typename T>
struct check_bit_set_t {
  uint32_t const* bitmaps{nullptr};
  T idx_first{};

  __device__ bool operator()(T idx) const
  {
    auto offset = idx - idx_first;
    auto mask   = uint32_t{1} << (offset % (sizeof(uint32_t) * 8));
    return (*(bitmaps + (offset / (sizeof(uint32_t) * 8))) & mask) > uint32_t{0};
  }
};

template <typename T>
struct check_in_range_t {
  T min{};  // inclusive
  T max{};  // exclusive

  __device__ bool operator()(T val) const { return (val >= min) && (val < max); }
};

template <typename T>
struct check_out_of_range_t {
  T min{};  // inclusive
  T max{};  // exclusive

  __device__ bool operator()(T val) const { return (val < min) || (val >= max); }
};

template <typename T>
struct strided_sum_t {
  T const* values{nullptr};
  size_t stride{0};
  size_t count{0};

  __device__ T operator()(size_t start_offset) const
  {
    T sum{0};
    for (size_t j = 0; j < count; ++j) {
      sum += values[start_offset + stride * j];
    }
    return sum;
  }
};

template <typename T>
struct shift_left_t {
  T offset{};

  __device__ T operator()(T input) const { return input - offset; }
};

template <typename T>
struct shift_right_t {
  T offset{};

  __device__ T operator()(T input) const { return input + offset; }
};

template <typename T>
struct multiplier_t {
  T multiplier{};

  __device__ T operator()(T input) const { return input * multiplier; }
};

template <typename T>
struct multiply_and_add_t {
  T multiplier{};
  T adder{};

  __device__ T operator()(T input) const { return input * multiplier + adder; }
};

template <typename T>
struct divider_t {
  T divisor{};

  __device__ T operator()(T input) const { return input / divisor; }
};

}  // namespace detail

}  // namespace cugraph

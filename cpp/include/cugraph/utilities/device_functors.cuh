/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cugraph/utilities/packed_bool_utils.hpp>

#include <thrust/iterator/iterator_traits.h>

#include <cstddef>

namespace cugraph {

namespace detail {

template <typename input_t, typename output_t>
struct typecast_t {
  __device__ output_t operator()(input_t val) const { return static_cast<output_t>(val); }
};

template <typename BoolIterator>
struct pack_bool_t {
  BoolIterator bool_first{};
  size_t num_bools{};

  __device__ uint32_t operator()(size_t i) const
  {
    auto first = i * packed_bools_per_word();
    auto last  = std::min((i + 1) * packed_bools_per_word(), num_bools);
    uint32_t ret{0};
    for (auto j = first; j < last; ++j) {
      if (*(bool_first + j)) {
        auto mask = packed_bool_mask(j);
        ret |= mask;
      }
    }
    return ret;
  }
};

template <typename PackedBoolIterator, typename T>
struct check_bit_set_t {
  PackedBoolIterator bitmap_first{};
  T idx_first{};

  static_assert(
    std::is_same_v<typename thrust::iterator_traits<PackedBoolIterator>::value_type, uint32_t>);

  __device__ bool operator()(T idx) const
  {
    auto offset = idx - idx_first;
    return static_cast<bool>(*(bitmap_first + packed_bool_offset(offset)) &
                             packed_bool_mask(offset));
  }
};

template <typename index_t, typename Iterator>
struct indirection_t {
  Iterator first{};

  __device__ typename thrust::iterator_traits<Iterator>::value_type operator()(index_t i) const
  {
    return *(first + i);
  }
};

template <typename index_t, typename Iterator>
struct indirection_if_idx_valid_t {
  using value_type = typename thrust::iterator_traits<Iterator>::value_type;
  Iterator first{};
  index_t invalid_idx{};
  value_type invalid_value{};

  __device__ value_type operator()(index_t i) const
  {
    return (i != invalid_idx) ? static_cast<value_type>(*(first + i)) : invalid_value;
  }
};

template <typename T>
struct is_equal_t {
  T compare{};

  __device__ bool operator()(T val) const { return val == compare; }
};

template <typename T>
struct is_not_equal_t {
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

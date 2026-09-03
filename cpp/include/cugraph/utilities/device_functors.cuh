/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/iterator_traits.h>

#include <cstddef>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename input_t, typename output_t>
struct typecast_t {
  __device__ output_t operator()(input_t val) const { return static_cast<output_t>(val); }
};

template <typename input_t, typename output_t>
struct converting_plus_t {
  __device__ output_t operator()(input_t lhs, input_t rhs) const
  {
    return static_cast<output_t>(lhs) + static_cast<output_t>(rhs);
  }
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
struct divide_and_indirection_t {
  Iterator first{};
  index_t divisor{};

  __device__ typename thrust::iterator_traits<Iterator>::value_type operator()(index_t i) const
  {
    return *(first + (i / divisor));
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

template <typename T, T compare>
struct is_equal_to_const_t {
  __device__ bool operator()(T val) const { return val == compare; }
};

template <typename T, T compare>
struct is_not_equal_to_const_t {
  __device__ bool operator()(T val) const { return val != compare; }
};

template <typename T>
struct invert_t {
  __device__ T operator()(T val) const { return -val; }
};

template <typename T>
struct is_less_than_t {
  T threshold{};

  __device__ bool operator()(T val) const { return val < threshold; }
};

template <typename T>
struct is_greater_than_or_equal_t {
  T threshold{};

  __device__ bool operator()(T val) const { return val >= threshold; }
};

template <typename T>
struct indirection_and_is_less_than_t {
  raft::device_span<T const> values{};
  T threshold{};

  __device__ bool operator()(size_t index) const { return values[index] < threshold; }
};

template <typename T>
struct indirection_and_is_greater_than_or_equal_t {
  raft::device_span<T const> values{};
  T threshold{};

  __device__ bool operator()(size_t index) const { return values[index] >= threshold; }
};

template <typename T>
struct adjacent_difference_t {
  raft::device_span<T const> offsets{};

  template <typename Index>
  __device__ T operator()(Index i) const
  {
    return offsets[i + 1] - offsets[i];
  }
};

struct clamped_subtract_t {
  size_t threshold{};

  template <typename Degree>
  __device__ size_t operator()(Degree value) const
  {
    auto d = static_cast<size_t>(value);
    return d > threshold ? (d - threshold) : size_t{0};
  }
};

template <typename T>
struct indirection_and_clamped_subtract_t {
  raft::device_span<T const> values{};
  size_t threshold{};

  __device__ size_t operator()(size_t i) const { return clamped_subtract_t{threshold}(values[i]); }
};

template <typename output_t = size_t>
struct segment_local_idx_t {
  raft::device_span<size_t const> segment_offsets{};

  __device__ output_t operator()(size_t i) const
  {
    auto segment_lasts = segment_offsets.subspan(1);
    auto idx           = cuda::std::distance(
      segment_lasts.begin(),
      thrust::upper_bound(thrust::seq, segment_lasts.begin(), segment_lasts.end(), i));
    return static_cast<output_t>(i - segment_offsets[idx]);
  }
};

template <typename index_t, typename T1, typename T2>
struct nested_indirection_t {
  raft::device_span<T1 const> first{};
  raft::device_span<T2 const> second{};

  __device__ T2 operator()(index_t i) const { return second[first[i]]; }
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

template <typename input_t, typename output_t>
struct modulo_t {
  input_t modulus{};

  __device__ output_t operator()(input_t input) const
  {
    return static_cast<output_t>(input % modulus);
  }
};

template <typename T>
struct segment_id_t {
  raft::device_span<T const> segment_lasts{};

  __device__ T operator()(T i) const
  {
    return static_cast<T>(cuda::std::distance(
      segment_lasts.begin(),
      thrust::upper_bound(thrust::seq, segment_lasts.begin(), segment_lasts.end(), i)));
  }
};

}  // namespace detail

}  // namespace CUGRAPH_EXPORT cugraph

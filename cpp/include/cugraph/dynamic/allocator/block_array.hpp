/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <cuda/std/tuple>
#include <cuda/stream>

#include <cstddef>
#include <type_traits>

namespace cugraph {
namespace dynamic {

/**
 * @brief GPU buffer split into fixed-size blocks.
 *
 * Free-block tracking lives in `block_allocator_t`. This object only stores the values.
 *
 * @tparam T Arithmetic type or cuda::std::tuple of arithmetic types.
 */
template <typename T>
class block_array_t {
 public:
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic_v<T>,
                "T must be an arithmetic type or cuda::std::tuple of arithmetic types.");

  using buffer_type = dataframe_buffer_type_t<T>;

  block_array_t(size_t block_capacity, size_t num_blocks, cuda::stream_ref stream)
    : block_capacity_(block_capacity),
      num_blocks_(num_blocks),
      block_storage_(allocate_dataframe_buffer<T>(num_blocks * block_capacity, stream))
  {
  }

  block_array_t()                                = delete;
  block_array_t(block_array_t const&)            = delete;
  block_array_t& operator=(block_array_t const&) = delete;

  block_array_t(block_array_t&&)            = default;
  block_array_t& operator=(block_array_t&&) = default;

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto column_data()
  {
    return block_storage_.data();
  }

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto column_data() const
  {
    return block_storage_.data();
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto column_data()
  {
    static_assert(I < thrust_tuple_size_or_one<U>::value);
    return std::get<I>(block_storage_).data();
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto column_data() const
  {
    static_assert(I < thrust_tuple_size_or_one<U>::value);
    return std::get<I>(block_storage_).data();
  }

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto block_data(size_t block_index)
  {
    return column_data() + block_index * block_capacity();
  }

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto block_data(size_t block_index) const
  {
    return column_data() + block_index * block_capacity();
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto block_data(size_t block_index)
  {
    return column_data<I>() + block_index * block_capacity();
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto block_data(size_t block_index) const
  {
    return column_data<I>() + block_index * block_capacity();
  }

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto element_data(size_t element_index)
  {
    return column_data() + element_index;
  }

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto element_data(size_t element_index) const
  {
    return column_data() + element_index;
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto element_data(size_t element_index)
  {
    return column_data<I>() + element_index;
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto element_data(size_t element_index) const
  {
    return column_data<I>() + element_index;
  }

  size_t block_capacity() const { return block_capacity_; }

  size_t num_blocks() const { return num_blocks_; }

  size_t num_elements() const { return size_dataframe_buffer(block_storage_); }

 private:
  size_t block_capacity_{0};
  size_t num_blocks_{0};
  buffer_type block_storage_;
};

}  // namespace dynamic
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dynamic/memory_manager/bit_tree.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/tuple>

#include <cstddef>
#include <type_traits>

namespace cugraph {
namespace dynamic {

/**
 * @brief Block array of fixed-size blocks with a host-side bit tree to track available blocks.
 *
 * @tparam T Arithmetic type or cuda::std::tuple of arithmetic types.
 */
template <typename T>
class block_array_t {
 public:
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic_v<T>,
                "T must be an arithmetic type or cuda::std::tuple of arithmetic types.");

  using buffer_type = dataframe_buffer_type_t<T>;

  block_array_t(size_t elements_per_block, size_t num_blocks, rmm::cuda_stream_view stream)
    : bit_tree_(elements_per_block, num_blocks),
      block_storage_(allocate_dataframe_buffer<T>(num_blocks * elements_per_block, stream))
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
    return column_data() + block_index * num_elements_per_block();
  }

  template <typename U = T, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  auto block_data(size_t block_index) const
  {
    return column_data() + block_index * num_elements_per_block();
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto block_data(size_t block_index)
  {
    return column_data<I>() + block_index * num_elements_per_block();
  }

  template <size_t I, typename U = T, std::enable_if_t<is_thrust_tuple_v<U>, int> = 0>
  auto block_data(size_t block_index) const
  {
    return column_data<I>() + block_index * num_elements_per_block();
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

  size_t insert() { return bit_tree_.insert(); }

  void remove(size_t block_index) { bit_tree_.remove(block_index); }

  size_t num_elements_per_block() const { return bit_tree_.num_elements_per_block(); }

  size_t num_blocks() const { return bit_tree_.num_blocks(); }

  size_t num_elements() const { return size_dataframe_buffer(block_storage_); }

  bool full() const { return bit_tree_.full(); }

  std::byte const* block_array_key() const
  {
    if constexpr (is_thrust_tuple_v<T>) {
      return reinterpret_cast<std::byte const*>(column_data<0>());
    } else {
      return reinterpret_cast<std::byte const*>(column_data());
    }
  }

 private:
  bit_tree_t bit_tree_;
  buffer_type block_storage_;
};

}  // namespace dynamic
}  // namespace cugraph

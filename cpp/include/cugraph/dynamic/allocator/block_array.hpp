/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <cuda/stream>

#include <cstddef>

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

  using buffer_type    = dataframe_buffer_type_t<T>;
  using iterator       = dataframe_buffer_iterator_type_t<T>;
  using const_iterator = dataframe_buffer_const_iterator_type_t<T>;

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

  iterator begin() { return get_dataframe_buffer_begin(block_storage_); }

  const_iterator cbegin() const { return get_dataframe_buffer_cbegin(block_storage_); }

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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dynamic/memory_manager/block_array.hpp>

#include <cuda/stream>

#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <vector>

namespace cugraph {
namespace dynamic {

/**
 * @brief Handle returned by block_array_manager_t::allocate() to locate an allocated block.
 *
 * `allocate(0)` returns `slab_id == invalid_slab_id`. Otherwise `num_elements_per_block` is
 * `size_t{1} << log_num_elements_per_block`.
 */
struct block_access_data_t {
  static constexpr size_t invalid_slab_id = std::numeric_limits<size_t>::max();

  size_t slab_id{invalid_slab_id};
  size_t block_index{0};
  size_t log_num_elements_per_block{0};
};

/**
 * @brief binning memory manager of block arrays (with # elements per block: 2^0, 2^1, ..., 2^n)
 *
 * @tparam T Arithmetic type or cuda::std::tuple of arithmetic types.
 * @param max_elements_per_block_array Maximum number of elements per block array. The maximum block
 * size can't exceed this value.
 */
template <typename T>
class block_array_manager_t {
 public:
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic_v<T>,
                "T must be an arithmetic type or cuda::std::tuple of arithmetic types.");

  static constexpr size_t num_bins_v = sizeof(size_t) * 8;

  explicit block_array_manager_t(size_t max_elements_per_block_array = (size_t{1} << 23))
    : max_elements_per_block_array_(std::bit_ceil(max_elements_per_block_array))
  {
    CUGRAPH_EXPECTS(std::has_single_bit(max_elements_per_block_array),
                    "Invalid input argument: max_elements_per_block_array must be a power of 2.");
  }

  block_array_manager_t(block_array_manager_t const&)            = delete;
  block_array_manager_t& operator=(block_array_manager_t const&) = delete;

  block_array_manager_t(block_array_manager_t&&)            = default;
  block_array_manager_t& operator=(block_array_manager_t&&) = default;

  block_access_data_t allocate(size_t num_elements_per_block, cuda::stream_ref stream)
  {
    if (num_elements_per_block == size_t{0}) { return block_access_data_t{}; }
    CUGRAPH_EXPECTS(std::has_single_bit(num_elements_per_block),
                    "Invalid input argument: num_elements_per_block must be a power of 2.");
    CUGRAPH_EXPECTS(
      num_elements_per_block <= max_elements_per_block_array_,
      "num_elements_per_block must be less than or equal to max_elements_per_block_array_.");

    auto const bin_index = std::bit_width(num_elements_per_block - size_t{1});
    auto& bin            = block_arrays_[bin_index];
    for (size_t slab_id = 0; slab_id < bin.size(); ++slab_id) {
      if (!bin[slab_id].full()) {
        auto const block_index = bin[slab_id].allocate();
        return {slab_id, block_index, bin[slab_id].log_block_capacity()};
      }
    }

    assert((max_elements_per_block_array_ % num_elements_per_block) == size_t{0});
    size_t const num_blocks = max_elements_per_block_array_ / num_elements_per_block;

    auto const slab_id = bin.size();
    bin.emplace_back(num_elements_per_block, num_blocks, stream);
    auto const block_index = bin.back().allocate();
    return {slab_id, block_index, bin.back().log_block_capacity()};
  }

  void release(block_access_data_t const& access_data)
  {
    CUGRAPH_EXPECTS(access_data.slab_id != block_access_data_t::invalid_slab_id,
                    "Cannot release an empty block_access_data_t.");
    auto const bin_index = access_data.log_num_elements_per_block;
    block_arrays_[bin_index].at(access_data.slab_id).release(access_data.block_index);
  }

  void release_all() noexcept
  {
    for (auto& bin : block_arrays_) {
      bin.clear();
    }
  }

 private:
  std::array<std::vector<block_array_t<T>>, num_bins_v> block_arrays_{};
  size_t max_elements_per_block_array_;
};

}  // namespace dynamic
}  // namespace cugraph

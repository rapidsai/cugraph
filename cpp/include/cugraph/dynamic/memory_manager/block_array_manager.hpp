/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dynamic/memory_manager/block_array.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <unordered_map>
#include <utility>

namespace cugraph {
namespace dynamic {

/**
 * @brief Handle returned by block_array_manager_t::insert() to locate an allocated block.
 */
struct block_access_data_t {
  std::byte const* block_array_key{nullptr};
  size_t block_index{0};
  size_t num_elements_per_block{0};
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

  block_access_data_t insert(size_t num_elements_per_block, rmm::cuda_stream_view stream)
  {
    CUGRAPH_EXPECTS(
      num_elements_per_block <= max_elements_per_block_array_,
      "num_elements_per_block must be less than or equal to max_elements_per_block_array_.");
    if (num_elements_per_block == size_t{0}) { return block_access_data_t{}; }

    auto const bin_index = std::bit_width(num_elements_per_block - size_t{1});
    for (auto& entry : block_arrays_[bin_index]) {
      if (!entry.second.full()) {
        auto block_index = entry.second.insert();
        return {entry.second.block_array_key(), block_index, entry.second.num_elements_per_block()};
      }
    }

    size_t const elements_per_block = std::bit_ceil(num_elements_per_block);

    assert((max_elements_per_block_array_ % elements_per_block) == size_t{0});
    size_t const num_blocks = max_elements_per_block_array_ / elements_per_block;

    block_array_t<T> new_block_array(elements_per_block, num_blocks, stream);
    size_t block_index = new_block_array.insert();

    auto block_array_key = new_block_array.block_array_key();
    block_arrays_[bin_index].emplace(block_array_key, std::move(new_block_array));
    return {block_array_key, block_index, elements_per_block};
  }

  void remove(block_access_data_t const& access_data)
  {
    CUGRAPH_EXPECTS(access_data.num_elements_per_block > size_t{0},
                    "num_elements_per_block must be greater than 0.");
    auto const bin_index = std::bit_width(access_data.num_elements_per_block - size_t{1});
    block_arrays_[bin_index].at(access_data.block_array_key).remove(access_data.block_index);
  }

  void remove_all() noexcept
  {
    for (auto& bin : block_arrays_) {
      bin.clear();
    }
  }

 private:
  std::array<std::unordered_map<std::byte const*, block_array_t<T>>,
             static_cast<size_t>(sizeof(T) * 8)>
    block_arrays_{};
  size_t max_elements_per_block_array_;
};

}  // namespace dynamic
}  // namespace cugraph

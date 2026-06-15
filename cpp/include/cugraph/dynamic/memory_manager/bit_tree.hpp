/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cugraph {
namespace dynamic {

/**
 * @brief Tracks which fixed-size blocks are free in a block array.
 *
 * A block array is a large fixed-capacity buffer split into equally-sized blocks. This host-side
 * bitmap records which blocks are available so insert() can hand out a free block quickly and
 * remove() can return one.
 *
 * Based on Hornet's Vec-Tree bitmap.
 */
class bit_tree_t {
 public:
  bit_tree_t(size_t elements_per_block, size_t num_blocks)
  {
    CUGRAPH_EXPECTS(std::has_single_bit(elements_per_block),
                    "Invalid input argument: elements_per_block must be a power of 2.");
    CUGRAPH_EXPECTS(std::has_single_bit(num_blocks),
                    "Invalid input argument: num_blocks must be a power of 2.");

    elements_per_block_ = elements_per_block;
    num_blocks_         = num_blocks;

    log_elements_per_block_ = std::bit_width(static_cast<size_t>(elements_per_block)) - size_t{1};
    auto log_num_blocks     = std::bit_width(num_blocks) - size_t{1};
    auto log_fanout         = std::bit_width(packed_bools_per_word()) - size_t{1};

    auto ceil_log_fanout = (log_num_blocks + log_fanout - size_t{1}) / log_fanout;
    num_internal_levels_ =
      ((ceil_log_fanout) >= size_t{1}) ? (ceil_log_fanout - size_t{1}) : size_t{0};

    // Internal summary bits cover the fanout tree above the leaf level. For L internal levels
    // and fanout F, the full tree has (F^(L+1) - 1) / (F - 1) nodes; drop the root summary bit.
    // e.g. 0 internal levels: 0, 1 internal levels: 32, 2 internal levels: 32*32 + 32, 3 internal
    // levels: 32*32*32 + 32*32 + 32
    auto constexpr fanout = packed_bools_per_word();
    auto fanout_power     = size_t{1};
    for (size_t i = 0; i <= num_internal_levels_; ++i) {
      fanout_power *= fanout;
    }
    internal_summary_bits_ = ((fanout_power - size_t{1}) / (fanout - size_t{1})) - size_t{1};
    internal_words_        = packed_bool_size(internal_summary_bits_);
    leaf_words_            = packed_bool_size(num_blocks_);
    array_.assign(internal_words_ + leaf_words_, packed_bool_full_mask());
  }

  bit_tree_t() = delete;

  size_t insert()
  {
    CUGRAPH_EXPECTS(num_occupied_blocks_ < num_blocks_, "Cannot insert into a full bit tree.");
    ++num_occupied_blocks_;

    size_t index = 0;
    for (size_t i = 0; i < num_internal_levels_; ++i) {
      auto pos = static_cast<size_t>(std::countr_zero(array_[index / packed_bools_per_word()]));
      index    = (index + pos + size_t{1}) * packed_bools_per_word();
    }
    index += static_cast<size_t>(std::countr_zero(array_[index / packed_bools_per_word()]));

    clear_available_bit(index);
    if (array_[index / packed_bools_per_word()] == packed_bool_empty_mask()) {
      // Walk up internal levels, clearing parent bits when a whole child group is full.
      for (size_t parent_index = index / packed_bools_per_word(); parent_index != size_t{0};
           parent_index /= packed_bools_per_word()) {
        --parent_index;
        clear_available_bit(parent_index);
        if (array_[parent_index / packed_bools_per_word()] != packed_bool_empty_mask()) { break; }
      }
    }

    auto block_index = index - internal_summary_bits_;
    return block_index;
  }

  void remove(size_t block_index)
  {
    CUGRAPH_EXPECTS(num_occupied_blocks_ != 0, "Cannot remove from an empty bit tree.");
    --num_occupied_blocks_;

    CUGRAPH_EXPECTS(block_index < num_blocks_,
                    "Invalid input argument: block_index is out of range.");
    CUGRAPH_EXPECTS(is_available_bit_clear(last_level_ptr(), block_index),
                    "Invalid input argument: block is not allocated.");

    set_available_bit(last_level_ptr(), block_index);
    block_index += internal_summary_bits_;

    // Walk up internal levels, setting parent bits until a parent already marks availability.
    for (size_t parent_index = block_index / packed_bools_per_word(); parent_index != size_t{0};
         parent_index /= packed_bools_per_word()) {
      --parent_index;
      bool parent_already_available =
        (array_[parent_index / packed_bools_per_word()] != packed_bool_empty_mask());
      set_available_bit(parent_index);
      if (parent_already_available) { break; }
    }
  }

  size_t num_occupied_blocks() const { return num_occupied_blocks_; }

  bool full() const { return num_occupied_blocks_ == num_blocks_; }

  size_t num_elements_per_block() const { return elements_per_block_; }

  size_t num_blocks() const { return num_blocks_; }

  size_t log_elements_per_block() const { return log_elements_per_block_; }

 private:
  uint32_t* last_level_ptr() { return array_.data() + internal_words_; }

  uint32_t const* last_level_ptr() const { return array_.data() + internal_words_; }

  void clear_available_bit(size_t bit_index)
  {
    array_[packed_bool_offset(bit_index)] &= ~packed_bool_mask(bit_index);
  }

  static void set_available_bit(uint32_t* array, size_t bit_index)
  {
    array[packed_bool_offset(bit_index)] |= packed_bool_mask(bit_index);
  }

  void set_available_bit(size_t bit_index) { set_available_bit(array_.data(), bit_index); }

  static bool is_available_bit_clear(uint32_t const* array, size_t bit_index)
  {
    return (array[packed_bool_offset(bit_index)] & packed_bool_mask(bit_index)) ==
           packed_bool_empty_mask();
  }

  size_t elements_per_block_{0};
  size_t log_elements_per_block_{0};
  size_t num_blocks_{0};
  size_t num_internal_levels_{0};
  size_t internal_summary_bits_{0};
  size_t internal_words_{0};
  size_t leaf_words_{0};

  std::vector<uint32_t> array_{};
  size_t num_occupied_blocks_{0};
};

}  // namespace dynamic
}  // namespace cugraph

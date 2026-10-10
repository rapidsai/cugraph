/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cugraph {
namespace dynamic {

/**
 * @brief Tracks which fixed-size blocks are free in a block array.
 *
 * A block array is a large fixed-capacity buffer split into equally-sized blocks. This host-side
 * bitmap records which blocks are available so allocate() can hand out a free block quickly and
 * release() can return one.
 *
 * Based on Hornet's Vec-Tree bitmap.
 */
class bit_tree_t {
 public:
  bit_tree_t(size_t block_capacity, size_t num_blocks)
  {
    CUGRAPH_EXPECTS(std::has_single_bit(block_capacity),
                    "Invalid input argument: block_capacity must be a power of 2.");
    CUGRAPH_EXPECTS(std::has_single_bit(num_blocks),
                    "Invalid input argument: num_blocks must be a power of 2.");

    block_capacity_ = block_capacity;
    num_blocks_     = num_blocks;

    log_block_capacity_ = std::bit_width(static_cast<size_t>(block_capacity)) - size_t{1};
    auto log_num_blocks = std::bit_width(num_blocks) - size_t{1};
    auto log_fanout     = std::bit_width(packed_bools_per_word()) - size_t{1};

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

  size_t allocate()
  {
    CUGRAPH_EXPECTS(num_occupied_blocks_ < num_blocks_, "Cannot allocate from a full bit tree.");
    ++num_occupied_blocks_;

    size_t index = first_available_leaf_bit_index();
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

  /**
   * @brief Bulk allocate `count` blocks.
   *
   * Fails when fewer than `count` blocks are free. The caller checks `num_available_blocks()` and
   * asks only for that many, then continues in another slab. A new slab filled completely skips
   * the bit walk. Taking every remaining free block starts at the first leaf word that still has
   * a free block, then clears the tree.
   */
  std::vector<size_t> allocate(size_t count)
  {
    CUGRAPH_EXPECTS(count <= num_available_blocks(),
                    "Cannot allocate from a bit tree with fewer free blocks than requested.");
    if (count == size_t{0}) { return {}; }

    if (num_occupied_blocks_ == size_t{0} && count == num_blocks_) {
      std::fill(array_.begin(), array_.end(), packed_bool_empty_mask());
      num_occupied_blocks_ = num_blocks_;
      std::vector<size_t> block_indices(num_blocks_);
      std::iota(block_indices.begin(), block_indices.end(), size_t{0});
      return block_indices;
    } else if (count == num_available_blocks()) {
      std::vector<size_t> block_indices;
      block_indices.reserve(count);
      auto const index      = first_available_leaf_bit_index();
      auto* const leaves    = last_level_ptr();
      auto const leaf_begin = (index - internal_summary_bits_) / packed_bools_per_word();
      for (size_t leaf_word = leaf_begin; leaf_word < leaf_words_; ++leaf_word) {
        auto word             = leaves[leaf_word];
        auto const block_base = leaf_word * packed_bools_per_word();
        while (word != packed_bool_empty_mask()) {
          auto const bit = static_cast<size_t>(std::countr_zero(word));
          if (block_base + bit >= num_blocks_) { break; }
          word &= ~packed_bool_mask(bit);
          block_indices.push_back(block_base + bit);
        }
      }
      std::fill(array_.begin(), array_.end(), packed_bool_empty_mask());
      num_occupied_blocks_ = num_blocks_;
      return block_indices;
    }

    std::vector<size_t> block_indices;
    block_indices.reserve(count);

    while (block_indices.size() < count) {
      auto const index       = first_available_leaf_bit_index();
      auto const word_offset = index / packed_bools_per_word();
      auto word              = array_[word_offset];
      auto const block_base  = index - internal_summary_bits_;
      auto const remaining   = count - block_indices.size();
      size_t taken           = 0;
      while (taken < remaining && word != packed_bool_empty_mask()) {
        auto const bit = static_cast<size_t>(std::countr_zero(word));
        if (block_base + bit >= num_blocks_) { break; }
        word &= ~packed_bool_mask(bit);
        block_indices.push_back(block_base + bit);
        ++taken;
      }
      array_[word_offset] = word;
      num_occupied_blocks_ += taken;

      if (word == packed_bool_empty_mask()) {
        for (size_t parent_index = index / packed_bools_per_word(); parent_index != size_t{0};
             parent_index /= packed_bools_per_word()) {
          --parent_index;
          clear_available_bit(parent_index);
          if (array_[parent_index / packed_bools_per_word()] != packed_bool_empty_mask()) { break; }
        }
      }
    }

    return block_indices;
  }

  void release(size_t block_index)
  {
    CUGRAPH_EXPECTS(num_occupied_blocks_ != 0, "Cannot release from an empty bit tree.");
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

  size_t num_available_blocks() const { return num_blocks_ - num_occupied_blocks_; }

  bool full() const { return num_occupied_blocks_ == num_blocks_; }

  size_t block_capacity() const { return block_capacity_; }

  size_t log_block_capacity() const { return log_block_capacity_; }

  size_t num_blocks() const { return num_blocks_; }

 private:
  uint32_t* last_level_ptr() { return array_.data() + internal_words_; }

  uint32_t const* last_level_ptr() const { return array_.data() + internal_words_; }

  // Bit index where the first leaf word with an available block begins.
  size_t first_available_leaf_bit_index() const
  {
    size_t index = 0;
    for (size_t level = 0; level < num_internal_levels_; ++level) {
      auto const pos =
        static_cast<size_t>(std::countr_zero(array_[index / packed_bools_per_word()]));
      index = (index + pos + size_t{1}) * packed_bools_per_word();
    }
    return index;
  }

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

  size_t block_capacity_{0};
  size_t log_block_capacity_{0};
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

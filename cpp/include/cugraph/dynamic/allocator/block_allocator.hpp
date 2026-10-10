/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dynamic/allocator/bit_tree.hpp>
#include <cugraph/utilities/error.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace cugraph {
namespace dynamic {

/**
 * @brief One allocated block inside a bin.
 */
struct block_location_t {
  size_t bin{0};
  size_t slab_id{0};
  size_t block_index{0};
};

/**
 * @brief Host-side free-block tracker.
 *
 * Owns one `bit_tree_t` per slab. Does not allocate device memory. A returned `slab_id` tells
 * `block_storage_t` to create or free that slab's buffer.
 *
 * A completely empty slab is freed. The tail slab is popped. A middle slab's id is kept on a
 * freed list and reused before the slab vector grows.
 */
class block_allocator_t {
 public:
  static constexpr size_t num_bins_v = sizeof(size_t) * 8;

  explicit block_allocator_t(size_t max_elements_per_slab = (size_t{1} << 23))
    : max_elements_per_slab_(max_elements_per_slab)
  {
    CUGRAPH_EXPECTS(std::has_single_bit(max_elements_per_slab),
                    "Invalid input argument: max_elements_per_slab must be a power of 2.");
  }

  block_allocator_t(block_allocator_t const&)            = delete;
  block_allocator_t& operator=(block_allocator_t const&) = delete;

  block_allocator_t(block_allocator_t&&)            = default;
  block_allocator_t& operator=(block_allocator_t&&) = default;

  /**
   * @brief Allocate one block.
   *
   * The optional `slab_id` has a value when this call allocates a slab buffer.
   */
  std::tuple<block_location_t, std::optional<size_t>> allocate(size_t block_capacity)
  {
    auto const bin = checked_bin(block_capacity);
    if (auto existing = allocate_from_live_slab(bin)) {
      return std::make_tuple(block_location_t{bin, std::get<0>(*existing), std::get<1>(*existing)},
                             std::optional<size_t>{std::nullopt});
    }

    auto const slab_id     = acquire_slab(bin, block_capacity);
    auto const block_index = bit_trees_[bin][slab_id].allocate();
    return std::make_tuple(block_location_t{bin, slab_id, block_index},
                           std::optional<size_t>{slab_id});
  }

  /**
   * @brief Allocate `count` blocks in one bin.
   *
   * The vector of slab ids has one entry per slab buffer allocated by this call.
   */
  std::tuple<std::vector<block_location_t>, std::vector<size_t>> allocate(size_t block_capacity,
                                                                          size_t count)
  {
    std::vector<block_location_t> blocks;
    std::vector<size_t> slab_ids;
    blocks.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto [block, slab_id] = allocate(block_capacity);
      blocks.push_back(block);
      if (slab_id.has_value()) { slab_ids.push_back(*slab_id); }
    }
    return std::make_tuple(std::move(blocks), std::move(slab_ids));
  }

  /**
   * @brief Free one block.
   *
   * The optional `slab_id` has a value when the block leaves its slab completely empty.
   */
  std::optional<size_t> release(block_location_t const& block)
  {
    CUGRAPH_EXPECTS(block.bin < num_bins_v, "Invalid input argument: bin is out of range.");
    auto& trees = bit_trees_[block.bin];
    CUGRAPH_EXPECTS(block.slab_id < trees.size(),
                    "Invalid input argument: slab_id is out of range.");
    CUGRAPH_EXPECTS(!is_slab_freed(block.bin, block.slab_id),
                    "Invalid input argument: slab buffer is already freed.");

    trees[block.slab_id].release(block.block_index);
    if (trees[block.slab_id].num_occupied_blocks() != size_t{0}) { return std::nullopt; }

    if (block.slab_id + size_t{1} == trees.size()) {
      pop_tail_and_empty_holes(block.bin);
    } else {
      freed_slab_ids_[block.bin].push_back(block.slab_id);
    }
    return block.slab_id;
  }

  size_t num_slabs(size_t bin) const
  {
    CUGRAPH_EXPECTS(bin < num_bins_v, "Invalid input argument: bin is out of range.");
    return bit_trees_[bin].size();
  }

 private:
  size_t checked_bin(size_t block_capacity) const
  {
    CUGRAPH_EXPECTS(block_capacity != size_t{0},
                    "Invalid input argument: block_capacity must be non-zero.");
    CUGRAPH_EXPECTS(std::has_single_bit(block_capacity),
                    "Invalid input argument: block_capacity must be a power of 2.");
    CUGRAPH_EXPECTS(block_capacity <= max_elements_per_slab_,
                    "block_capacity must be less than or equal to max_elements_per_slab.");
    return std::bit_width(block_capacity) - size_t{1};
  }

  bool is_slab_freed(size_t bin, size_t slab_id) const
  {
    auto const& freed = freed_slab_ids_[bin];
    return std::find(freed.begin(), freed.end(), slab_id) != freed.end();
  }

  std::optional<std::tuple<size_t, size_t>> allocate_from_live_slab(size_t bin)
  {
    auto& trees = bit_trees_[bin];
    for (size_t slab_id = 0; slab_id < trees.size(); ++slab_id) {
      if (is_slab_freed(bin, slab_id) || trees[slab_id].full()) { continue; }
      return std::make_tuple(slab_id, trees[slab_id].allocate());
    }
    return std::nullopt;
  }

  size_t acquire_slab(size_t bin, size_t block_capacity)
  {
    auto& freed = freed_slab_ids_[bin];
    if (!freed.empty()) {
      auto const slab_id = freed.back();
      freed.pop_back();
      return slab_id;
    }

    auto const num_blocks = max_elements_per_slab_ / block_capacity;
    bit_trees_[bin].emplace_back(block_capacity, num_blocks);
    return bit_trees_[bin].size() - size_t{1};
  }

  void pop_tail_and_empty_holes(size_t bin)
  {
    auto& trees = bit_trees_[bin];
    auto& freed = freed_slab_ids_[bin];
    trees.pop_back();
    while (!trees.empty()) {
      auto const slab_id = trees.size() - size_t{1};
      auto const it      = std::find(freed.begin(), freed.end(), slab_id);
      if (it == freed.end()) { break; }
      freed.erase(it);
      trees.pop_back();
    }
  }

  std::array<std::vector<bit_tree_t>, num_bins_v> bit_trees_{};
  std::array<std::vector<size_t>, num_bins_v> freed_slab_ids_{};
  size_t max_elements_per_slab_;
};

}  // namespace dynamic
}  // namespace cugraph

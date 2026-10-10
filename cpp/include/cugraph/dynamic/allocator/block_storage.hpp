/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dynamic/allocator/block_array.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda/stream>

#include <array>
#include <bit>
#include <cstddef>
#include <optional>
#include <vector>

namespace cugraph {
namespace dynamic {

/**
 * @brief Passive slab buffers indexed by `(bin, slab_id)` from `block_allocator_t`.
 *
 * An allocate or release that stays inside existing slabs does not change this object. A returned
 * slab id creates a buffer, or frees one. A middle slab keeps its id after its buffer is freed. The
 * tail, and any already-freed slabs that become the tail, are popped.
 *
 * @tparam T Arithmetic type or cuda::std::tuple of arithmetic types.
 */
template <typename T>
class block_storage_t {
 public:
  static constexpr size_t num_bins_v = sizeof(size_t) * 8;

  explicit block_storage_t(size_t max_elements_per_slab = (size_t{1} << 23))
    : max_elements_per_slab_(max_elements_per_slab)
  {
    CUGRAPH_EXPECTS(std::has_single_bit(max_elements_per_slab),
                    "Invalid input argument: max_elements_per_slab must be a power of 2.");
  }

  block_storage_t(block_storage_t const&)            = delete;
  block_storage_t& operator=(block_storage_t const&) = delete;

  block_storage_t(block_storage_t&&)            = default;
  block_storage_t& operator=(block_storage_t&&) = default;

  void allocate_slab(size_t bin, size_t slab_id, cuda::stream_ref stream)
  {
    CUGRAPH_EXPECTS(bin < num_bins_v, "Invalid input argument: bin is out of range.");
    auto const block_capacity = size_t{1} << bin;
    CUGRAPH_EXPECTS(block_capacity <= max_elements_per_slab_,
                    "Invalid input argument: bin exceeds max_elements_per_slab.");
    auto const num_blocks = max_elements_per_slab_ / block_capacity;

    auto& slabs = slabs_[bin];
    if (slab_id == slabs.size()) {
      slabs.emplace_back(std::in_place, block_capacity, num_blocks, stream);
      return;
    }

    CUGRAPH_EXPECTS(slab_id < slabs.size(), "Invalid input argument: slab_id is out of range.");
    CUGRAPH_EXPECTS(!slabs[slab_id].has_value(),
                    "Invalid input argument: slab buffer is already allocated.");
    slabs[slab_id].emplace(block_capacity, num_blocks, stream);
  }

  void release_slab(size_t bin, size_t slab_id)
  {
    CUGRAPH_EXPECTS(bin < num_bins_v, "Invalid input argument: bin is out of range.");
    auto& slabs = slabs_[bin];
    CUGRAPH_EXPECTS(slab_id < slabs.size(), "Invalid input argument: slab_id is out of range.");
    CUGRAPH_EXPECTS(slabs[slab_id].has_value(),
                    "Invalid input argument: slab buffer is already freed.");

    if (slab_id + size_t{1} == slabs.size()) {
      slabs.pop_back();
      while (!slabs.empty() && !slabs.back().has_value()) {
        slabs.pop_back();
      }
      return;
    }

    slabs[slab_id].reset();
  }

  size_t num_slabs(size_t bin) const
  {
    CUGRAPH_EXPECTS(bin < num_bins_v, "Invalid input argument: bin is out of range.");
    return slabs_[bin].size();
  }

  bool contains(size_t bin, size_t slab_id) const
  {
    CUGRAPH_EXPECTS(bin < num_bins_v, "Invalid input argument: bin is out of range.");
    return slab_id < slabs_[bin].size() && slabs_[bin][slab_id].has_value();
  }

  block_array_t<T>& block_array(size_t bin, size_t slab_id)
  {
    CUGRAPH_EXPECTS(contains(bin, slab_id),
                    "Invalid input argument: slab buffer is not allocated.");
    return *slabs_[bin][slab_id];
  }

  block_array_t<T> const& block_array(size_t bin, size_t slab_id) const
  {
    CUGRAPH_EXPECTS(contains(bin, slab_id),
                    "Invalid input argument: slab buffer is not allocated.");
    return *slabs_[bin][slab_id];
  }

 private:
  std::array<std::vector<std::optional<block_array_t<T>>>, num_bins_v> slabs_{};
  size_t max_elements_per_slab_;
};

}  // namespace dynamic
}  // namespace cugraph

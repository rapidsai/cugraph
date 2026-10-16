/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/dynamic/allocator/block_array.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/device_span.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <array>
#include <bit>
#include <cstddef>
#include <vector>

namespace cugraph {
namespace dynamic {

/**
 * @brief Passive slab buffers indexed by `(bin, slab_id)` from `block_allocator_t`.
 *
 * An allocate or release that stays inside existing slabs does not change this object. A returned
 * slab id creates a buffer, or frees one. A middle slab keeps its id and is replaced by a
 * zero-capacity `block_array_t`, which releases the old buffer. The tail, and any already-freed
 * slabs that become the tail, are popped.
 *
 * @tparam T Arithmetic type or cuda::std::tuple of arithmetic types.
 */
template <typename T>
class block_storage_t {
 public:
  static constexpr size_t num_bins_v = sizeof(size_t) * 8;

  using iterator = typename block_array_t<T>::iterator;

  explicit block_storage_t(size_t max_elements_per_slab, cuda::stream_ref stream)
    : slab_iterator_ptrs_(num_bins_v, stream), max_elements_per_slab_(max_elements_per_slab)
  {
    CUGRAPH_EXPECTS(std::has_single_bit(max_elements_per_slab_),
                    "Invalid input argument: max_elements_per_slab must be a power of 2.");
    slab_iterators_.reserve(num_bins_v);
    for (size_t bin = 0; bin < num_bins_v; ++bin) {
      slab_iterators_.emplace_back(size_t{0}, stream);
    }
    std::vector<iterator const*> empty_bins(num_bins_v, nullptr);
    raft::update_device(slab_iterator_ptrs_.data(), empty_bins.data(), empty_bins.size(), stream);
    stream.sync();
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

    auto& slabs         = slabs_[bin];
    auto const appended = slab_id == slabs.size();
    if (appended) {
      slabs.emplace_back(block_capacity, num_blocks, stream);
    } else {
      CUGRAPH_EXPECTS(slab_id < slabs.size(), "Invalid input argument: slab_id is out of range.");
      CUGRAPH_EXPECTS(slabs[slab_id].block_capacity() == size_t{0},
                      "Invalid input argument: slab buffer is already allocated.");
      slabs[slab_id] = block_array_t<T>(block_capacity, num_blocks, stream);
    }

    if (appended) { slab_iterators_[bin].resize(slabs.size(), stream); }
    set_slab_iterator(bin, slab_id, slabs[slab_id].begin(), stream);
    if (appended) { publish_slab_iterators(bin, stream); }
  }

  void release_slab(size_t bin, size_t slab_id, cuda::stream_ref stream)
  {
    CUGRAPH_EXPECTS(bin < num_bins_v, "Invalid input argument: bin is out of range.");
    auto& slabs = slabs_[bin];
    CUGRAPH_EXPECTS(slab_id < slabs.size(), "Invalid input argument: slab_id is out of range.");
    CUGRAPH_EXPECTS(slabs[slab_id].block_capacity() != size_t{0},
                    "Invalid input argument: slab buffer is already freed.");

    if (slab_id + size_t{1} == slabs.size()) {
      auto new_size = slab_id;
      while (new_size > size_t{0} && slabs[new_size - size_t{1}].block_capacity() == size_t{0}) {
        --new_size;
      }
      slab_iterators_[bin].resize(new_size, stream);
      publish_slab_iterators(bin, stream);
      while (slabs.size() > new_size) {
        slabs.pop_back();
      }
      return;
    }

    set_slab_iterator(bin, slab_id, iterator{}, stream);
    slabs[slab_id] = block_array_t<T>(size_t{0}, size_t{0}, stream);
  }

  auto slab_iterator_ptrs() const
  {
    return raft::device_span<iterator const* const>(slab_iterator_ptrs_.data(),
                                                    slab_iterator_ptrs_.size());
  }

 private:
  void set_slab_iterator(size_t bin,
                         size_t slab_id,
                         iterator slab_iterator,
                         cuda::stream_ref stream)
  {
    raft::update_device(slab_iterators_[bin].data() + slab_id, &slab_iterator, size_t{1}, stream);
    stream.sync();
  }

  void publish_slab_iterators(size_t bin, cuda::stream_ref stream)
  {
    iterator const* slab_iterator_table{
      slab_iterators_[bin].is_empty() ? nullptr : slab_iterators_[bin].data()};
    raft::update_device(slab_iterator_ptrs_.data() + bin, &slab_iterator_table, size_t{1}, stream);
    stream.sync();
  }

  std::array<std::vector<block_array_t<T>>, num_bins_v> slabs_{};
  std::vector<rmm::device_uvector<iterator>> slab_iterators_{};  // size: num_bins_v
  rmm::device_uvector<iterator const*> slab_iterator_ptrs_;
  size_t max_elements_per_slab_{};
};

}  // namespace dynamic
}  // namespace cugraph

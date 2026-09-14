/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

template <typename key_t, typename value_t>
void device_segmented_sort_pairs(raft::handle_t const& handle,
                                 raft::device_span<key_t const> keys_in,
                                 raft::device_span<key_t> keys_out,
                                 raft::device_span<value_t const> values_in,
                                 raft::device_span<value_t> values_out,
                                 raft::device_span<size_t const> begin_offsets,
                                 raft::device_span<size_t const> end_offsets)
{
  assert(keys_in.size() == keys_out.size());
  assert(values_in.size() == values_out.size());
  assert(keys_in.size() == values_in.size());
  assert(begin_offsets.size() == end_offsets.size());

  size_t tmp_storage_bytes{0};
  cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                      tmp_storage_bytes,
                                      keys_in.data(),
                                      keys_out.data(),
                                      values_in.data(),
                                      values_out.data(),
                                      keys_in.size(),
                                      begin_offsets.size(),
                                      begin_offsets.data(),
                                      end_offsets.data(),
                                      handle.get_stream().get());
  rmm::device_uvector<std::byte> d_tmp_storage(tmp_storage_bytes, handle.get_stream());
  cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                      tmp_storage_bytes,
                                      keys_in.data(),
                                      keys_out.data(),
                                      values_in.data(),
                                      values_out.data(),
                                      keys_in.size(),
                                      begin_offsets.size(),
                                      begin_offsets.data(),
                                      end_offsets.data(),
                                      handle.get_stream().get());
}

// offsets is a CSR offset array of size num_segments + 1
template <typename key_t, typename value_t>
void device_segmented_sort_pairs(raft::handle_t const& handle,
                                 raft::device_span<key_t const> keys_in,
                                 raft::device_span<key_t> keys_out,
                                 raft::device_span<value_t const> values_in,
                                 raft::device_span<value_t> values_out,
                                 raft::device_span<size_t const> offsets)
{
  assert(offsets.size() >= size_t{1});
  device_segmented_sort_pairs(
    handle,
    keys_in,
    keys_out,
    values_in,
    values_out,
    raft::device_span<size_t const>{offsets.data(), offsets.size() - 1},
    raft::device_span<size_t const>{offsets.data() + 1, offsets.size() - 1});
}

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph

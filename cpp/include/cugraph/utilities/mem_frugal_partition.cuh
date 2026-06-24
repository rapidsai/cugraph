/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/mask_utils.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

// Use roughly half temporary buffer than thrust::partition (if first & second partition sizes are
// comparable). This also uses multiple smaller allocations than one single allocation (thrust::sort
// does this) of the same aggregate size if the input iterators are the zip iterators (this is more
// favorable to the pool allocator).
template <typename ValueIterator, typename ValueToGroupIdOp>
ValueIterator mem_frugal_partition(
  ValueIterator value_first,
  ValueIterator value_last,
  ValueToGroupIdOp value_to_group_id_op,
  int pivot,  // group id less than pivot goes to the first partition
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto num_elements = static_cast<size_t>(cuda::std::distance(value_first, value_last));
  auto [marked_count, marked_entries] =
    mark_entries(num_elements,
                 cuda::proclaim_return_type<bool>(
                   [value_first, value_to_group_id_op, pivot] __device__(size_t i) {
                     return value_to_group_id_op(*(value_first + i)) < pivot;
                   }),
                 stream_view,
                 large_buffer_type);

  auto const first_size      = marked_count;
  auto const second_size     = num_elements - first_size;
  uint32_t const* mask_first = marked_entries.data();

  partition_by_mask(
    value_first, value_last, mask_first, first_size, second_size, stream_view, large_buffer_type);

  return value_first + first_size;
}

// Use roughly half temporary buffer than thrust::partition (if first & second partition sizes are
// comparable). This also uses multiple smaller allocations than one single allocation (thrust::sort
// does this) of the same aggregate size if the input iterators are the zip iterators (this is more
// favorable to the pool allocator).
template <typename KeyIterator, typename ValueIterator, typename KeyToGroupIdOp>
std::tuple<KeyIterator, ValueIterator> mem_frugal_partition(
  KeyIterator key_first,
  KeyIterator key_last,
  ValueIterator value_first,
  KeyToGroupIdOp key_to_group_id_op,
  int pivot,  // group Id less than pivot goes to the first partition
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto num_elements = static_cast<size_t>(cuda::std::distance(key_first, key_last));
  auto [marked_count, marked_entries] = mark_entries(
    num_elements,
    [key_first, key_to_group_id_op, pivot] __device__(size_t i) {
      return key_to_group_id_op(*(key_first + i)) < pivot;
    },
    stream_view,
    large_buffer_type);

  auto const first_size      = marked_count;
  auto const second_size     = num_elements - first_size;
  uint32_t const* mask_first = marked_entries.data();

  partition_by_mask(
    key_first, key_last, mask_first, first_size, second_size, stream_view, large_buffer_type);

  partition_by_mask(value_first,
                    value_first + num_elements,
                    mask_first,
                    first_size,
                    second_size,
                    stream_view,
                    large_buffer_type);

  return std::make_tuple(key_first + first_size, value_first + first_size);
}

}  // namespace detail

}  // namespace CUGRAPH_EXPORT cugraph

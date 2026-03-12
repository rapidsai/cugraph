/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <iostream>

namespace cugraph {
namespace detail {

void swap_marked_and_unmarked_entries(raft::handle_t const& handle,
                                      raft::device_span<uint32_t> flags,
                                      size_t num_entries)
{
  uint32_t full_mask    = packed_bool_full_mask();
  uint32_t partial_mask = packed_bool_partial_mask(num_entries % packed_bools_per_word());

  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(size_t{0}),
                   thrust::make_counting_iterator(flags.size()),
                   [flags, full_mask, partial_mask] __device__(size_t idx) {
                     if (idx == (flags.size() - 1)) {
                       flags[idx] ^= partial_mask;
                     } else {
                       flags[idx] ^= full_mask;
                     }
                   });
}

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <thrust/tabulate.h>

namespace cugraph {
namespace detail {

void invert_flags(raft::handle_t const& handle,
                  raft::device_span<uint32_t> flags,
                  size_t num_entries)
{
  uint32_t full_mask    = packed_bool_full_mask();
  uint32_t partial_mask = packed_bool_partial_mask(num_entries % packed_bools_per_word());

  auto const n_words = flags.size();
  thrust::tabulate(handle.get_thrust_policy(),
                   flags.begin(),
                   flags.end(),
                   [p = flags.data(), n_words, full_mask, partial_mask] __device__(size_t idx) {
                     uint32_t const word = p[idx];
                     return (idx + 1 == n_words) ? (word ^ partial_mask) : (word ^ full_mask);
                   });
}

}  // namespace detail
}  // namespace cugraph

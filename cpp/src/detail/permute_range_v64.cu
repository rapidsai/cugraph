/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/permute_range.cuh"

namespace cugraph {

namespace detail {

template rmm::device_uvector<int64_t> permute_range(raft::handle_t const& handle,
                                                    raft::random::RngState& rng_state,
                                                    int64_t local_range_start,
                                                    int64_t local_range_size,
                                                    bool multi_gpu,
                                                    bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph

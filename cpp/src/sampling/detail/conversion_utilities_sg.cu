/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/conversion_utilities_impl.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<int32_t> flatten_label_map(
  raft::handle_t const& handle,
  std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>
    label_to_output_comm_rank);

}  // namespace detail
}  // namespace cugraph

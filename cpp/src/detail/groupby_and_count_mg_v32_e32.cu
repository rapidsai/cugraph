/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby_and_count.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  raft::device_span<int32_t> edgelist_majors,
  raft::device_span<int32_t> edgelist_minors,
  raft::host_span<cugraph::arithmetic_device_span_t> edgelist_properties,
  bool groupby_and_counts_local_partition,
  std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace detail
}  // namespace cugraph

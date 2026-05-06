/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/gather_sampled_properties_impl.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views);

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/deduplicate_edges_by_minor_impl.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    arithmetic_device_uvector_t,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor<int64_t, int64_t, true>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  rmm::device_uvector<int64_t>&&,
  rmm::device_uvector<int64_t>&&,
  arithmetic_device_uvector_t&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  bool);

}  // namespace detail
}  // namespace cugraph

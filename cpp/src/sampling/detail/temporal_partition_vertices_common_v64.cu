/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "temporal_partition_vertices_impl.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_partition_vertices(raft::handle_t const& handle,
                            raft::device_span<int64_t const> vertices,
                            raft::device_span<int32_t const> vertex_times,
                            std::optional<raft::device_span<int32_t const>> vertex_labels);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_partition_vertices(raft::handle_t const& handle,
                            raft::device_span<int64_t const> vertices,
                            raft::device_span<int64_t const> vertex_times,
                            std::optional<raft::device_span<int32_t const>> vertex_labels);

}  // namespace detail
}  // namespace cugraph

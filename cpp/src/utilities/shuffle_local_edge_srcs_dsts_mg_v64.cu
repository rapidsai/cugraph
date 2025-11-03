/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shuffle_local_edge_srcs_dsts.cuh"

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_local_edge_srcs(raft::handle_t const& handle,
                        rmm::device_uvector<int64_t>&& edge_srcs,
                        std::vector<cugraph::arithmetic_device_uvector_t>&& edge_src_properties,
                        raft::host_span<int64_t const> vertex_partition_range_lasts,
                        bool store_transposed);

template std::tuple<rmm::device_uvector<int64_t>, std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_local_edge_dsts(raft::handle_t const& handle,
                        rmm::device_uvector<int64_t>&& edge_dsts,
                        std::vector<cugraph::arithmetic_device_uvector_t>&& edge_dst_properties,
                        raft::host_span<int64_t const> vertex_partition_range_lasts,
                        bool store_transposed);

}  // namespace cugraph

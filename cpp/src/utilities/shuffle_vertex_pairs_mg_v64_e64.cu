/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shuffle_vertex_pairs.cuh"

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<cugraph::arithmetic_device_uvector_t>,
                    std::vector<size_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<int64_t>&& edge_srcs,
                  rmm::device_uvector<int64_t>&& edge_dsts,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<cugraph::arithmetic_device_uvector_t>,
                    std::vector<size_t>>
shuffle_int_edges(raft::handle_t const& handle,
                  rmm::device_uvector<int64_t>&& majors,
                  rmm::device_uvector<int64_t>&& minors,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  raft::host_span<int64_t const> vertex_partition_range_lasts,
                  std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace cugraph

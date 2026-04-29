/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shuffle_vertex_pairs.cuh"

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<int32_t>&& edge_srcs,
                  rmm::device_uvector<int32_t>&& edge_dsts,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_int_edges(raft::handle_t const& handle,
                  rmm::device_uvector<int32_t>&& majors,
                  rmm::device_uvector<int32_t>&& minors,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  raft::host_span<int32_t const> vertex_partition_range_lasts,
                  std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/detail/leiden/leiden_csr_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT leiden_result_t
leiden_csr(raft::handle_t const& handle,
           uint32_t seed,
           raft::device_span<int32_t const> offsets,
           raft::device_span<int32_t const> indices,
           std::optional<raft::device_span<float const>> weights,
           raft::device_span<int32_t> labels,
           leiden_params_t const& params,
           leiden_layout_options_t const& options);

template CUGRAPH_EXPORT leiden_result_t
leiden_csr(raft::handle_t const& handle,
           uint32_t seed,
           raft::device_span<int32_t const> offsets,
           raft::device_span<int32_t const> indices,
           std::optional<raft::device_span<double const>> weights,
           raft::device_span<int32_t> labels,
           leiden_params_t const& params,
           leiden_layout_options_t const& options);

template CUGRAPH_EXPORT leiden_result_t
leiden_coo<int32_t, int32_t, float>(raft::handle_t const& handle,
                                    uint32_t seed,
                                    int32_t num_vertices,
                                    rmm::device_uvector<int32_t>&& srcs,
                                    rmm::device_uvector<int32_t>&& dsts,
                                    std::optional<rmm::device_uvector<float>>&& weights,
                                    raft::device_span<int32_t> labels,
                                    leiden_params_t const& params,
                                    leiden_layout_options_t const& options);

template CUGRAPH_EXPORT leiden_result_t
leiden_coo<int32_t, int32_t, double>(raft::handle_t const& handle,
                                     uint32_t seed,
                                     int32_t num_vertices,
                                     rmm::device_uvector<int32_t>&& srcs,
                                     rmm::device_uvector<int32_t>&& dsts,
                                     std::optional<rmm::device_uvector<double>>&& weights,
                                     raft::device_span<int32_t> labels,
                                     leiden_params_t const& params,
                                     leiden_layout_options_t const& options);

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/collect_comm_wrapper.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<int64_t> device_allgatherv(raft::handle_t const& handle,
                                                        raft::comms::comms_t const& comm,
                                                        raft::device_span<int64_t const> d_input);

template rmm::device_uvector<double> device_allgatherv(raft::handle_t const& handle,
                                                       raft::comms::comms_t const& comm,
                                                       raft::device_span<double const> d_input);

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "utilities/collect_comm.cuh"

namespace cugraph {
namespace detail {

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::comms::comms_t const& comm,
                                         raft::device_span<T const> d_input)
{
  auto gathered_v = cugraph::device_allgatherv(handle, comm, d_input);

  return gathered_v;
}

}  // namespace detail
}  // namespace cugraph

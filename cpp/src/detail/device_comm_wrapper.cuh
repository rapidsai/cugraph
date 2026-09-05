/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <numeric>
#include <vector>

namespace cugraph {
namespace detail {

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::comms::comms_t const& comms,
                                         raft::device_span<T const> d_input)
{
  auto rx_sizes = cugraph::host_scalar_allgather(comms, d_input.size(), handle.get_stream());
  std::vector<size_t> rx_displs(static_cast<size_t>(comms.get_size()));
  std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);

  rmm::device_uvector<T> gathered_v(std::reduce(rx_sizes.begin(), rx_sizes.end()),
                                    handle.get_stream());

  cugraph::device_allgatherv(comms,
                             d_input.data(),
                             gathered_v.data(),
                             raft::host_span<size_t const>(rx_sizes.data(), rx_sizes.size()),
                             raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                             handle.get_stream());

  return gathered_v;
}

}  // namespace detail
}  // namespace cugraph

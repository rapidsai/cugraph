/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/call_thrust_is_sorted.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/sort.h>

#include <limits>

namespace cugraph {

void call_thrust_is_sorted(raft::handle_t const& handle,
                           raft::device_span<size_t const> subgraph_offsets)
{
  raft::print_device_vector(
    "subgraph_offsets", subgraph_offsets.data(), subgraph_offsets.size(), std::cout);
  CUGRAPH_EXPECTS(
    thrust::is_sorted(handle.get_thrust_policy(), subgraph_offsets.begin(), subgraph_offsets.end()),
    "Invalid input argument: subgraph_offsets is not sorted.");
}

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/thrust_wrapper.hpp"

#include <cugraph/call_thrust_is_sorted.hpp>

int main(int argc, char** argv)
{
  raft::handle_t handle{};

  rmm::device_uvector<int32_t> srcs(10, handle.get_stream());
  cugraph::test::translate_vertex_ids(handle, srcs, int32_t{0});

  std::vector<size_t> h_subgraph_offsets({0, 10});
  rmm::device_uvector<size_t> subgraph_offsets(2, handle.get_stream());
  raft::update_device(subgraph_offsets.data(),
                      h_subgraph_offsets.data(),
                      h_subgraph_offsets.size(),
                      handle.get_stream());
  cugraph::call_thrust_is_sorted(
    handle, raft::device_span<size_t const>{subgraph_offsets.data(), subgraph_offsets.size()});
  return 0;
}

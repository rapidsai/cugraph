/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sampling/random_walks_check.cuh"

namespace cugraph {
namespace test {

template void random_walks_validate(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>>,
  rmm::device_uvector<int32_t>&& d_start,
  rmm::device_uvector<int32_t>&& d_vertices,
  std::optional<rmm::device_uvector<float>>&& d_weights,
  size_t max_length);

}  // namespace test
}  // namespace cugraph

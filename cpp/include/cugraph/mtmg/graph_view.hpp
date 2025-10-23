/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/graph_view.hpp>
#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Graph view for each GPU
 */
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_view_t : public detail::device_shared_wrapper_t<
                       cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>> {
 public:
  /**
   * @brief Get the vertex_partition_view for this graph
   */
  vertex_partition_view_t<vertex_t, multi_gpu> get_vertex_partition_view(
    cugraph::mtmg::handle_t const& handle) const
  {
    return this->get(handle).local_vertex_partition_view();
  }

  /**
   * @brief Get the vertex_partition_view for this graph
   */
  raft::host_span<vertex_t const> get_vertex_partition_range_lasts(
    cugraph::mtmg::handle_t const& handle) const
  {
    return this->get(handle).vertex_partition_range_lasts();
  }
};

}  // namespace mtmg
}  // namespace cugraph

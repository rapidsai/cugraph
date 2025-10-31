/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_device_span_tuple.hpp>
#include <cugraph/mtmg/graph_view.hpp>
#include <cugraph/mtmg/handle.hpp>
#include <cugraph/mtmg/renumber_map.hpp>

#include <optional>

namespace cugraph {
namespace mtmg {

/**
 * @brief An MTMG device span for referencing a vertex pair result
 */
template <typename vertex_t, typename result_t>
class vertex_pair_result_view_t
  : public detail::device_shared_device_span_tuple_t<vertex_t, vertex_t, result_t> {
  using parent_t = detail::device_shared_device_span_tuple_t<vertex_t, vertex_t, result_t>;

 public:
  vertex_pair_result_view_t(parent_t&& other) : parent_t{std::move(other)} {}

  /**
   * @brief Gather results from specified vertices
   */
  template <bool multi_gpu>
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             rmm::device_uvector<result_t>>
  gather(handle_t const& handle,
         raft::device_span<vertex_t const> vertices,
         raft::host_span<vertex_t const> vertex_partition_range_lasts,
         cugraph::vertex_partition_view_t<vertex_t, multi_gpu> vertex_partition_view,
         std::optional<cugraph::mtmg::renumber_map_view_t<vertex_t>>& renumber_map_view);
};

}  // namespace mtmg
}  // namespace cugraph

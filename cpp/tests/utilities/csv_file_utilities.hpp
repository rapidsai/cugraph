/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/graph.hpp>
#include <cugraph/large_buffer_manager.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <numeric>
#include <optional>
#include <string>
#include <type_traits>

namespace cugraph {
namespace test {

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           bool>
read_edgelist_from_csv_file(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool store_transposed,
  bool multi_gpu,
  bool shuffle                                                = true,
  std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
  std::optional<large_buffer_type_t> large_edge_buffer_type   = std::nullopt);

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<cugraph::edge_property_t<edge_t, weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
read_graph_from_csv_file(raft::handle_t const& handle,
                         std::string const& graph_file_full_path,
                         bool test_weighted,
                         bool renumber,
                         std::optional<large_buffer_type_t> large_vertex_buffer_type = std::nullopt,
                         std::optional<large_buffer_type_t> large_edge_buffer_type = std::nullopt);

}  // namespace test
}  // namespace cugraph

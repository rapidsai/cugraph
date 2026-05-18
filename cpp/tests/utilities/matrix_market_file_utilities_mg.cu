/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/matrix_market_file_utilities_impl.cuh"

namespace cugraph {
namespace test {

template std::tuple<cugraph::graph_t<int32_t, int32_t, false, true>,
                    std::optional<cugraph::edge_property_t<int32_t, float>>,
                    std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int32_t, int32_t, true, true>,
                    std::optional<cugraph::edge_property_t<int32_t, float>>,
                    std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int32_t, int32_t, false, true>,
                    std::optional<cugraph::edge_property_t<int32_t, double>>,
                    std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int32_t, int32_t, true, true>,
                    std::optional<cugraph::edge_property_t<int32_t, double>>,
                    std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int64_t, int64_t, false, true>,
                    std::optional<cugraph::edge_property_t<int64_t, float>>,
                    std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int64_t, int64_t, true, true>,
                    std::optional<cugraph::edge_property_t<int64_t, float>>,
                    std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int64_t, int64_t, false, true>,
                    std::optional<cugraph::edge_property_t<int64_t, double>>,
                    std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

template std::tuple<cugraph::graph_t<int64_t, int64_t, true, true>,
                    std::optional<cugraph::edge_property_t<int64_t, double>>,
                    std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type);

}  // namespace test
}  // namespace cugraph

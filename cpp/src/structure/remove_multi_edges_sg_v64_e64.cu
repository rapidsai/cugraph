/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/remove_multi_edges_impl.cuh"

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int64_t>&& edgelist_srcs,
                   rmm::device_uvector<int64_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<float>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_start_times,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_end_times,
                   bool keep_min_value_edge,
                   std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int64_t>&& edgelist_srcs,
                   rmm::device_uvector<int64_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<double>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_start_times,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_end_times,
                   bool keep_min_value_edge,
                   std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int64_t>&& edgelist_srcs,
                   rmm::device_uvector<int64_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<float>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_start_times,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_end_times,
                   bool keep_min_value_edge,
                   std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int64_t>&& edgelist_srcs,
                   rmm::device_uvector<int64_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<double>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_start_times,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_end_times,
                   bool keep_min_value_edge,
                   std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<std::vector<rmm::device_uvector<int64_t>>,
                    std::vector<rmm::device_uvector<int64_t>>,
                    std::optional<std::vector<rmm::device_uvector<float>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>>
remove_multi_edges(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<float>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_end_times,
  bool keep_min_value_edge,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<std::vector<rmm::device_uvector<int64_t>>,
                    std::vector<rmm::device_uvector<int64_t>>,
                    std::optional<std::vector<rmm::device_uvector<double>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>>
remove_multi_edges(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<double>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_end_times,
  bool keep_min_value_edge,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<std::vector<rmm::device_uvector<int64_t>>,
                    std::vector<rmm::device_uvector<int64_t>>,
                    std::optional<std::vector<rmm::device_uvector<float>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>>
remove_multi_edges(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<float>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_end_times,
  bool keep_min_value_edge,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<std::vector<rmm::device_uvector<int64_t>>,
                    std::vector<rmm::device_uvector<int64_t>>,
                    std::optional<std::vector<rmm::device_uvector<double>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int32_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>,
                    std::optional<std::vector<rmm::device_uvector<int64_t>>>>
remove_multi_edges(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<double>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<int32_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<int64_t>>>&& edgelist_edge_end_times,
  bool keep_min_value_edge,
  std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace cugraph

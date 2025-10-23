/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/decompress_to_edgelist_impl.cuh"

namespace cugraph {

// SG instantiation

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
decompress_to_edgelist<int32_t, int32_t, float, int32_t, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
decompress_to_edgelist<int32_t, int32_t, float, int32_t, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
decompress_to_edgelist<int32_t, int32_t, double, int32_t, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
decompress_to_edgelist<int32_t, int32_t, double, int32_t, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check);

}  // namespace cugraph

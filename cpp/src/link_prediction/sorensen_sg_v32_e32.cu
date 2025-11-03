/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "link_prediction/sorensen_impl.cuh"

namespace cugraph {

template rmm::device_uvector<float> sorensen_coefficients(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>> vertex_pairs,
  bool do_expensive_check);

template rmm::device_uvector<double> sorensen_coefficients(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>> vertex_pairs,
  bool do_expensive_check);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
  sorensen_all_pairs_coefficients(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    std::optional<size_t> topk,
    bool do_expensive_check);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
  sorensen_all_pairs_coefficients(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    std::optional<size_t> topk,
    bool do_expensive_check);

}  // namespace cugraph

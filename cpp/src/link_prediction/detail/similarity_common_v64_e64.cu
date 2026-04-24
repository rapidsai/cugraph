/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "link_prediction/detail/similarity_impl.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<float> similarity<int64_t, int64_t, float, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::tuple<raft::device_span<int64_t const>, raft::device_span<int64_t const>> vertex_pairs,
  coefficient_t coeff,
  bool do_expensive_check);

template rmm::device_uvector<double> similarity<int64_t, int64_t, double, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::tuple<raft::device_span<int64_t const>, raft::device_span<int64_t const>> vertex_pairs,
  coefficient_t coeff,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
all_pairs_similarity<int64_t, int64_t, float, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> vertices,
  std::optional<size_t> topk,
  coefficient_t coeff,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
all_pairs_similarity<int64_t, int64_t, double, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> vertices,
  std::optional<size_t> topk,
  coefficient_t coeff,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph

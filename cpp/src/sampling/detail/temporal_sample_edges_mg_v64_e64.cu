/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sample_edges.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_edges(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                      raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views,
                      edge_property_view_t<int64_t, int32_t const*> edge_start_time_view,
                      std::optional<edge_arithmetic_property_view_t<int64_t>> edge_type_view,
                      std::optional<edge_arithmetic_property_view_t<int64_t>> edge_bias_view,
                      raft::device_span<int64_t const> active_majors,
                      raft::device_span<int32_t const> active_major_times,
                      std::optional<raft::device_span<int32_t const>> active_major_labels,
                      raft::host_span<size_t const> Ks,
                      bool with_replacement,
                      temporal_sampling_comparison_t temporal_sampling_comparison,
                      std::optional<int32_t> window_start,
                      std::optional<int32_t> window_end);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_edges(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                      raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views,
                      edge_property_view_t<int64_t, int64_t const*> edge_start_time_view,
                      std::optional<edge_arithmetic_property_view_t<int64_t>> edge_type_view,
                      std::optional<edge_arithmetic_property_view_t<int64_t>> edge_bias_view,
                      raft::device_span<int64_t const> active_majors,
                      raft::device_span<int64_t const> active_major_times,
                      std::optional<raft::device_span<int32_t const>> active_major_labels,
                      raft::host_span<size_t const> Ks,
                      bool with_replacement,
                      temporal_sampling_comparison_t temporal_sampling_comparison,
                      std::optional<int64_t> window_start,
                      std::optional<int64_t> window_end);

}  // namespace detail
}  // namespace cugraph

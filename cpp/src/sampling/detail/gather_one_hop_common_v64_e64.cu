/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gather_one_hop_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

using vertex_t = int64_t;
using edge_t   = int64_t;

#define CUGRAPH_INSTANTIATE_GATHER_ONE_HOP(multi_gpu)                             \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,               \
                                     rmm::device_uvector<vertex_t>,               \
                                     arithmetic_device_uvector_t,                 \
                                     std::optional<rmm::device_uvector<int32_t>>> \
  gather_one_hop_edgelist(                                                        \
    raft::handle_t const& handle,                                                 \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,           \
    size_t number_of_edge_properties,                                             \
    std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,   \
    raft::device_span<vertex_t const> active_majors,                              \
    std::optional<raft::device_span<int32_t const>> active_major_labels,          \
    std::optional<raft::device_span<bool const>> gather_flags,                    \
    bool do_expensive_check);                                                     \
                                                                                  \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,               \
                                     rmm::device_uvector<vertex_t>,               \
                                     arithmetic_device_uvector_t,                 \
                                     std::optional<rmm::device_uvector<int32_t>>, \
                                     rmm::device_uvector<vertex_t>,               \
                                     std::optional<rmm::device_uvector<int32_t>>> \
  gather_one_hop_edgelist_to_unvisited_neighbors(                                 \
    raft::handle_t const& handle,                                                 \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,           \
    size_t number_of_edge_properties,                                             \
    std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,   \
    raft::device_span<vertex_t const> active_majors,                              \
    std::optional<raft::device_span<int32_t const>> active_major_labels,          \
    std::optional<raft::device_span<bool const>> gather_flags,                    \
    rmm::device_uvector<vertex_t>&& visited_minors,                               \
    std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,           \
    bool do_expensive_check)

#define CUGRAPH_INSTANTIATE_TEMPORAL_GATHER_ONE_HOP(time_stamp_t, multi_gpu)         \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                  \
                                     rmm::device_uvector<vertex_t>,                  \
                                     arithmetic_device_uvector_t,                    \
                                     std::optional<rmm::device_uvector<int32_t>>>    \
  temporal_gather_one_hop_edgelist(                                                  \
    raft::handle_t const& handle,                                                    \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,              \
    edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,                \
    std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,      \
    raft::device_span<vertex_t const> active_majors,                                 \
    std::optional<raft::device_span<time_stamp_t const>> active_major_window_starts, \
    std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,   \
    std::optional<raft::device_span<int32_t const>> active_major_labels,             \
    std::optional<raft::device_span<bool const>> gather_flags,                       \
    temporal_sampling_comparison_t temporal_sampling_comparison,                     \
    bool fixed_window,                                                               \
    bool do_expensive_check);                                                        \
                                                                                     \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                  \
                                     rmm::device_uvector<vertex_t>,                  \
                                     std::vector<arithmetic_device_uvector_t>,       \
                                     std::optional<rmm::device_uvector<int32_t>>,    \
                                     rmm::device_uvector<vertex_t>,                  \
                                     std::optional<rmm::device_uvector<int32_t>>>    \
  temporal_gather_one_hop_edgelist_to_unvisited_neighbors(                           \
    raft::handle_t const& handle,                                                    \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,              \
    raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,    \
    edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,                \
    std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,      \
    raft::device_span<vertex_t const> active_majors,                                 \
    std::optional<raft::device_span<time_stamp_t const>> active_major_window_starts, \
    std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,   \
    std::optional<raft::device_span<int32_t const>> active_major_labels,             \
    std::optional<raft::device_span<bool const>> gather_flags,                       \
    rmm::device_uvector<vertex_t>&& visited_minors,                                  \
    std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,              \
    temporal_sampling_comparison_t temporal_sampling_comparison,                     \
    bool fixed_window,                                                               \
    bool do_expensive_check)

CUGRAPH_INSTANTIATE_GATHER_ONE_HOP(false);
CUGRAPH_INSTANTIATE_GATHER_ONE_HOP(true);
CUGRAPH_INSTANTIATE_TEMPORAL_GATHER_ONE_HOP(int32_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_GATHER_ONE_HOP(int32_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_GATHER_ONE_HOP(int64_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_GATHER_ONE_HOP(int64_t, true);

#undef CUGRAPH_INSTANTIATE_TEMPORAL_GATHER_ONE_HOP
#undef CUGRAPH_INSTANTIATE_GATHER_ONE_HOP

}  // namespace detail
}  // namespace cugraph

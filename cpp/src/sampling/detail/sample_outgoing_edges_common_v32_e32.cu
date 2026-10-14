/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sample_outgoing_edges_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

using vertex_t = int32_t;
using edge_t   = int32_t;

#define CUGRAPH_INSTANTIATE_SAMPLE_OUTGOING_EDGES(multi_gpu)                             \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                      \
                                     rmm::device_uvector<vertex_t>,                      \
                                     arithmetic_device_uvector_t,                        \
                                     std::optional<rmm::device_uvector<int32_t>>>        \
  sample_outgoing_edges<vertex_t, edge_t, multi_gpu>(                                    \
    raft::handle_t const& handle,                                                        \
    raft::random::RngState& rng_state,                                                   \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                  \
    bool has_output_edge_properties,                                                     \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,               \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,               \
    cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false> const& key_bucket_view, \
    raft::host_span<size_t const> Ks,                                                    \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                 \
    bool with_replacement)

#define CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_OUTGOING_EDGES(tag_t, multi_gpu)             \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                       \
                                     rmm::device_uvector<vertex_t>,                       \
                                     arithmetic_device_uvector_t,                         \
                                     std::optional<rmm::device_uvector<int32_t>>,         \
                                     rmm::device_uvector<vertex_t>,                       \
                                     std::optional<rmm::device_uvector<int32_t>>>         \
  sample_unvisited_outgoing_edges<vertex_t, edge_t, tag_t, multi_gpu>(                    \
    raft::handle_t const& handle,                                                         \
    raft::random::RngState& rng_state,                                                    \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                   \
    bool has_output_edge_properties,                                                      \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,                \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,                \
    cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false> const& key_bucket_view, \
    rmm::device_uvector<vertex_t>&& visited_minors,                                       \
    std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,                   \
    raft::host_span<size_t const> Ks,                                                     \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                  \
    bool with_replacement,                                                                \
    no_temporal_params_t temporal_params)

#define CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(                           \
  tag_t, time_stamp_t, multi_gpu)                                                               \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                             \
                                     rmm::device_uvector<vertex_t>,                             \
                                     arithmetic_device_uvector_t,                               \
                                     std::optional<rmm::device_uvector<int32_t>>,               \
                                     rmm::device_uvector<vertex_t>,                             \
                                     std::optional<rmm::device_uvector<int32_t>>>               \
  sample_unvisited_outgoing_edges<vertex_t,                                                     \
                                  edge_t,                                                       \
                                  tag_t,                                                        \
                                  multi_gpu,                                                    \
                                  temporal_unvisited_params_t<vertex_t, edge_t, time_stamp_t>>( \
    raft::handle_t const& handle,                                                               \
    raft::random::RngState& rng_state,                                                          \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                         \
    bool has_output_edge_properties,                                                            \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,                      \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,                      \
    cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false> const& key_bucket_view,       \
    rmm::device_uvector<vertex_t>&& visited_minors,                                             \
    std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,                         \
    raft::host_span<size_t const> Ks,                                                           \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                        \
    bool with_replacement,                                                                      \
    temporal_unvisited_params_t<vertex_t, edge_t, time_stamp_t> temporal_params)

CUGRAPH_INSTANTIATE_SAMPLE_OUTGOING_EDGES(false);
CUGRAPH_INSTANTIATE_SAMPLE_OUTGOING_EDGES(true);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_OUTGOING_EDGES(void, false);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_OUTGOING_EDGES(void, true);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_OUTGOING_EDGES(int32_t, false);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_OUTGOING_EDGES(int32_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(void, int32_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(void, int32_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(void, int64_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(void, int64_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(int32_t, int32_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(int32_t, int32_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(int32_t, int64_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES(int32_t, int64_t, true);

#undef CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_UNVISITED_OUTGOING_EDGES
#undef CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_OUTGOING_EDGES
#undef CUGRAPH_INSTANTIATE_SAMPLE_OUTGOING_EDGES

}  // namespace detail
}  // namespace cugraph

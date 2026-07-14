/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sample_edges_one_property_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

using vertex_t           = int64_t;
using edge_t             = int64_t;
using multi_index_view_t = edge_multi_index_property_view_t<edge_t, vertex_t>;

#define CUGRAPH_INSTANTIATE_SAMPLE_WITH_ONE_PROPERTY(multi_gpu)                          \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                      \
                                     rmm::device_uvector<vertex_t>,                      \
                                     arithmetic_device_uvector_t,                        \
                                     std::optional<rmm::device_uvector<int32_t>>>        \
  sample_with_one_property<vertex_t, edge_t, edge_dummy_property_view_t, multi_gpu>(     \
    raft::handle_t const& handle,                                                        \
    raft::random::RngState& rng_state,                                                   \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                  \
    edge_dummy_property_view_t edge_property_view,                                       \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,               \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,               \
    cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false> const& key_bucket_view, \
    raft::host_span<size_t const> Ks,                                                    \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                 \
    bool with_replacement);                                                              \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                      \
                                     rmm::device_uvector<vertex_t>,                      \
                                     arithmetic_device_uvector_t,                        \
                                     std::optional<rmm::device_uvector<int32_t>>>        \
  sample_with_one_property<vertex_t, edge_t, multi_index_view_t, multi_gpu>(             \
    raft::handle_t const& handle,                                                        \
    raft::random::RngState& rng_state,                                                   \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                  \
    multi_index_view_t edge_property_view,                                               \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,               \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,               \
    cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false> const& key_bucket_view, \
    raft::host_span<size_t const> Ks,                                                    \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                 \
    bool with_replacement)

#define CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_WITH_ONE_PROPERTY(tag_t, multi_gpu)              \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                           \
                                     rmm::device_uvector<vertex_t>,                           \
                                     arithmetic_device_uvector_t,                             \
                                     std::optional<rmm::device_uvector<int32_t>>,             \
                                     rmm::device_uvector<vertex_t>,                           \
                                     std::optional<rmm::device_uvector<int32_t>>>             \
  sample_unvisited_with_one_property<vertex_t,                                                \
                                     edge_t,                                                  \
                                     tag_t,                                                   \
                                     edge_dummy_property_view_t,                              \
                                     multi_gpu>(                                              \
    raft::handle_t const& handle,                                                             \
    raft::random::RngState& rng_state,                                                        \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                       \
    edge_dummy_property_view_t edge_property_view,                                            \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,                    \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,                    \
    cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false> const& key_bucket_view,     \
    rmm::device_uvector<vertex_t>&& visited_minors,                                           \
    std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,                       \
    raft::host_span<size_t const> Ks,                                                         \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                      \
    bool with_replacement);                                                                   \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                           \
                                     rmm::device_uvector<vertex_t>,                           \
                                     arithmetic_device_uvector_t,                             \
                                     std::optional<rmm::device_uvector<int32_t>>,             \
                                     rmm::device_uvector<vertex_t>,                           \
                                     std::optional<rmm::device_uvector<int32_t>>>             \
  sample_unvisited_with_one_property<vertex_t, edge_t, tag_t, multi_index_view_t, multi_gpu>( \
    raft::handle_t const& handle,                                                             \
    raft::random::RngState& rng_state,                                                        \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                       \
    multi_index_view_t edge_property_view,                                                    \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,                    \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,                    \
    cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false> const& key_bucket_view,     \
    rmm::device_uvector<vertex_t>&& visited_minors,                                           \
    std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,                       \
    raft::host_span<size_t const> Ks,                                                         \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                      \
    bool with_replacement)

#define CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_WITH_ONE_PROPERTY(time_stamp_t, multi_gpu)           \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                              \
                                     rmm::device_uvector<vertex_t>,                              \
                                     arithmetic_device_uvector_t,                                \
                                     std::optional<rmm::device_uvector<int32_t>>>                \
  temporal_sample_with_one_property<vertex_t,                                                    \
                                    edge_t,                                                      \
                                    edge_dummy_property_view_t,                                  \
                                    time_stamp_t,                                                \
                                    multi_gpu>(                                                  \
    raft::handle_t const& handle,                                                                \
    raft::random::RngState& rng_state,                                                           \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                          \
    edge_dummy_property_view_t edge_property_view,                                               \
    edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,                            \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,                       \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,                       \
    cugraph::key_bucket_view_t<vertex_t, time_stamp_t, multi_gpu, false> const& key_bucket_view, \
    raft::host_span<size_t const> Ks,                                                            \
    bool with_replacement,                                                                       \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                         \
    raft::device_span<vertex_t const> active_majors,                                             \
    raft::device_span<time_stamp_t const> active_major_times,                                    \
    std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,               \
    temporal_sampling_comparison_t temporal_sampling_comparison);                                \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,                              \
                                     rmm::device_uvector<vertex_t>,                              \
                                     arithmetic_device_uvector_t,                                \
                                     std::optional<rmm::device_uvector<int32_t>>>                \
  temporal_sample_with_one_property<vertex_t,                                                    \
                                    edge_t,                                                      \
                                    multi_index_view_t,                                          \
                                    time_stamp_t,                                                \
                                    multi_gpu>(                                                  \
    raft::handle_t const& handle,                                                                \
    raft::random::RngState& rng_state,                                                           \
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,                          \
    multi_index_view_t edge_property_view,                                                       \
    edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,                            \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,                       \
    std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,                       \
    cugraph::key_bucket_view_t<vertex_t, time_stamp_t, multi_gpu, false> const& key_bucket_view, \
    raft::host_span<size_t const> Ks,                                                            \
    bool with_replacement,                                                                       \
    std::optional<raft::device_span<int32_t const>> active_major_labels,                         \
    raft::device_span<vertex_t const> active_majors,                                             \
    raft::device_span<time_stamp_t const> active_major_times,                                    \
    std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,               \
    temporal_sampling_comparison_t temporal_sampling_comparison)

CUGRAPH_INSTANTIATE_SAMPLE_WITH_ONE_PROPERTY(false);
CUGRAPH_INSTANTIATE_SAMPLE_WITH_ONE_PROPERTY(true);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_WITH_ONE_PROPERTY(void, false);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_WITH_ONE_PROPERTY(void, true);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_WITH_ONE_PROPERTY(int32_t, false);
CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_WITH_ONE_PROPERTY(int32_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_WITH_ONE_PROPERTY(int32_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_WITH_ONE_PROPERTY(int32_t, true);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_WITH_ONE_PROPERTY(int64_t, false);
CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_WITH_ONE_PROPERTY(int64_t, true);

#undef CUGRAPH_INSTANTIATE_TEMPORAL_SAMPLE_WITH_ONE_PROPERTY
#undef CUGRAPH_INSTANTIATE_SAMPLE_UNVISITED_WITH_ONE_PROPERTY
#undef CUGRAPH_INSTANTIATE_SAMPLE_WITH_ONE_PROPERTY

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/random_walks_impl.cuh"

#include <cugraph/algorithms.hpp>

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<float>>>
uniform_random_walks(raft::handle_t const& handle,
                     raft::random::RngState& rng_state,
                     graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                     std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                     raft::device_span<int64_t const> start_vertices,
                     size_t max_length);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<double>>>
uniform_random_walks(raft::handle_t const& handle,
                     raft::random::RngState& rng_state,
                     graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                     std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                     raft::device_span<int64_t const> start_vertices,
                     size_t max_length);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<float>>>
biased_random_walks(raft::handle_t const& handle,
                    raft::random::RngState& rng_state,
                    graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                    edge_property_view_t<int64_t, float const*> edge_weight_view,
                    raft::device_span<int64_t const> start_vertices,
                    size_t max_length);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<double>>>
biased_random_walks(raft::handle_t const& handle,
                    raft::random::RngState& rng_state,
                    graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                    edge_property_view_t<int64_t, double const*> edge_weight_view,
                    raft::device_span<int64_t const> start_vertices,
                    size_t max_length);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<float>>>
node2vec_random_walks(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                      std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
                      raft::device_span<int64_t const> start_vertices,
                      size_t max_length,
                      float p,
                      float q);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<double>>>
node2vec_random_walks(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                      std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
                      raft::device_span<int64_t const> start_vertices,
                      size_t max_length,
                      double p,
                      double q);

}  // namespace cugraph

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

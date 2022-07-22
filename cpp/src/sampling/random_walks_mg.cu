/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>

namespace cugraph {

// FIXME: Temporarily here until random_walks_impl.cuh is ready with the real implementation
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                     raft::device_span<vertex_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed)
{
  CUGRAPH_FAIL("Not Implemented");
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                    raft::device_span<vertex_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed)
{
  CUGRAPH_FAIL("Not Implemented");
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                      raft::device_span<vertex_t const> start_vertices,
                      size_t max_length,
                      weight_t p,
                      weight_t q,
                      uint64_t seed)
{
  CUGRAPH_FAIL("Not Implemented");
}

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                     raft::device_span<int32_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                     raft::device_span<int32_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<float>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                     raft::device_span<int64_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                     raft::device_span<int32_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                     raft::device_span<int32_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<double>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                     raft::device_span<int64_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                    raft::device_span<int32_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                    raft::device_span<int32_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<float>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                    raft::device_span<int64_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                    raft::device_span<int32_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                    raft::device_span<int32_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<double>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                    raft::device_span<int64_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                      raft::device_span<int32_t const> start_vertices,
                      size_t max_length,
                      float p,
                      float q,
                      uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                      raft::device_span<int32_t const> start_vertices,
                      size_t max_length,
                      float p,
                      float q,
                      uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<float>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                      raft::device_span<int64_t const> start_vertices,
                      size_t max_length,
                      float p,
                      float q,
                      uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                      raft::device_span<int32_t const> start_vertices,
                      size_t max_length,
                      double p,
                      double q,
                      uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                      raft::device_span<int32_t const> start_vertices,
                      size_t max_length,
                      double p,
                      double q,
                      uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, std::optional<rmm::device_uvector<double>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                      raft::device_span<int64_t const> start_vertices,
                      size_t max_length,
                      double p,
                      double q,
                      uint64_t seed);

}  // namespace cugraph

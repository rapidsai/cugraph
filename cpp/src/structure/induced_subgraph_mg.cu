/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <structure/induced_subgraph_impl.cuh>

namespace cugraph {

// MG instantiation

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int32_t, float, true, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int32_t, double, true, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int64_t, float, true, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int64_t, double, true, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int32_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int64_t, int64_t, float, true, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int64_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int64_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int64_t, int64_t, double, true, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int64_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_induced_subgraphs(raft::handle_t const& handle,
                          graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                          size_t const* subgraph_offsets,
                          int64_t const* subgraph_vertices,
                          size_t num_subgraphs,
                          bool do_expensive_check);

}  // namespace cugraph

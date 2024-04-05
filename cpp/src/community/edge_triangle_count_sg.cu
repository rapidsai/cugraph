/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "community/edge_triangle_count_impl.cuh"

namespace cugraph {

// SG instantiation
template rmm::device_uvector<int32_t> edge_triangle_count(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  raft::device_span<int32_t> edgelist_srcs,
  raft::device_span<int32_t> edgelist_dsts);

template rmm::device_uvector<int64_t> edge_triangle_count(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  raft::device_span<int32_t> edgelist_srcs,
  raft::device_span<int32_t> edgelist_dsts);

template rmm::device_uvector<int64_t> edge_triangle_count(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  raft::device_span<int64_t> edgelist_srcs,
  raft::device_span<int64_t> edgelist_dsts);

}  // namespace cugraph

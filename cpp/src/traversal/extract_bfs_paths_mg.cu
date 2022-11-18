/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <traversal/extract_bfs_paths_impl.cuh>

namespace cugraph {

// MG instantiation

template std::tuple<rmm::device_uvector<int32_t>, int32_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  int32_t const* distances,
  int32_t const* predecessors,
  int32_t const* destinations,
  size_t n_destinations);

template std::tuple<rmm::device_uvector<int32_t>, int32_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  int32_t const* distances,
  int32_t const* predecessors,
  int32_t const* destinations,
  size_t n_destinations);

template std::tuple<rmm::device_uvector<int64_t>, int64_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  int64_t const* distances,
  int64_t const* predecessors,
  int64_t const* destinations,
  size_t n_destinations);

}  // namespace cugraph

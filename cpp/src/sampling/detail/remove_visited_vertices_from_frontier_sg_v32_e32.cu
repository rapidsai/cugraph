/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "sampling/detail/remove_visited_vertices_from_frontier.cuh"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<int32_t>>>
remove_visited_vertices_from_frontier(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& frontier_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& frontier_vertex_labels,
  raft::device_span<int32_t const> vertices_used_as_source,
  std::optional<raft::device_span<int32_t const>> vertex_labels_used_as_source);

}  // namespace detail
}  // namespace cugraph

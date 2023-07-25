/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <sampling/detail/prepare_next_frontier_impl.cuh>

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<std::tuple<rmm::device_uvector<int32_t>,
                                             std::optional<rmm::device_uvector<int32_t>>>>>
prepare_next_frontier(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> sampled_src_vertices,
  std::optional<raft::device_span<int32_t const>> sampled_src_vertex_labels,
  raft::device_span<int32_t const> sampled_dst_vertices,
  std::optional<raft::device_span<int32_t const>> sampled_dst_vertex_labels,
  std::optional<std::tuple<rmm::device_uvector<int32_t>,
                           std::optional<rmm::device_uvector<int32_t>>>>&& vertex_used_as_source,
  vertex_partition_view_t<int32_t, true> vertex_partition,
  std::vector<int32_t> const& vertex_partition_range_lasts,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<std::tuple<rmm::device_uvector<int64_t>,
                                             std::optional<rmm::device_uvector<int32_t>>>>>
prepare_next_frontier(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> sampled_src_vertices,
  std::optional<raft::device_span<int32_t const>> sampled_src_vertex_labels,
  raft::device_span<int64_t const> sampled_dst_vertices,
  std::optional<raft::device_span<int32_t const>> sampled_dst_vertex_labels,
  std::optional<std::tuple<rmm::device_uvector<int64_t>,
                           std::optional<rmm::device_uvector<int32_t>>>>&& vertex_used_as_source,
  vertex_partition_view_t<int64_t, true> vertex_partition,
  std::vector<int64_t> const& vertex_partition_range_lasts,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph

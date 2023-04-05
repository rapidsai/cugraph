/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <community/detail/mis_impl.cuh>

namespace cugraph {
namespace detail {
template rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& decision_graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view);

template rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& decision_graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view);

template rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& decision_graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view);

template rmm::device_uvector<int32_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& decision_graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view);

template rmm::device_uvector<int64_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& decision_graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view);

template rmm::device_uvector<int64_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& decision_graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view);

}  // namespace detail
}  // namespace cugraph

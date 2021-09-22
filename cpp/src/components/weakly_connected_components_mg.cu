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

#include <components/weakly_connected_components_impl.cuh>

namespace cugraph {

// MG instantiations
template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
  int32_t* components,
  bool do_expensive_check);

template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
  int32_t* components,
  bool do_expensive_check);

template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
  int32_t* components,
  bool do_expensive_check);

template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
  int32_t* components,
  bool do_expensive_check);

template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
  int64_t* components,
  bool do_expensive_check);

template void weakly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
  int64_t* components,
  bool do_expensive_check);

}  // namespace cugraph

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

// Andrei Schaffer, aschaffer@nvidia.com
//
#include "graph_bcast.cuh"

namespace cugraph {
namespace broadcast {
// Manual template instantiations (EIDir's):
//
template graph_t<int32_t, int32_t, float, false, false> graph_broadcast(
  raft::handle_t const& handle, graph_t<int32_t, int32_t, float, false, false>* graph_ptr);

template graph_t<int32_t, int64_t, float, false, false> graph_broadcast(
  raft::handle_t const& handle, graph_t<int32_t, int64_t, float, false, false>* graph_ptr);

template graph_t<int64_t, int64_t, float, false, false> graph_broadcast(
  raft::handle_t const& handle, graph_t<int64_t, int64_t, float, false, false>* graph_ptr);

template graph_t<int32_t, int32_t, double, false, false> graph_broadcast(
  raft::handle_t const& handle, graph_t<int32_t, int32_t, double, false, false>* graph_ptr);

template graph_t<int32_t, int64_t, double, false, false> graph_broadcast(
  raft::handle_t const& handle, graph_t<int32_t, int64_t, double, false, false>* graph_ptr);

template graph_t<int64_t, int64_t, double, false, false> graph_broadcast(
  raft::handle_t const& handle, graph_t<int64_t, int64_t, double, false, false>* graph_ptr);

}  // namespace broadcast
}  // namespace cugraph

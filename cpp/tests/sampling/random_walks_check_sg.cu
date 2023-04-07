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
#include <sampling/random_walks_check.cuh>

namespace cugraph {
namespace test {

template void random_walks_validate(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>>,
  rmm::device_uvector<int32_t>&& d_start,
  rmm::device_uvector<int32_t>&& d_vertices,
  std::optional<rmm::device_uvector<float>>&& d_weights,
  size_t max_length);

}  // namespace test
}  // namespace cugraph

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

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace test {

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>> unrenumber_kv_pairs(
  raft::handle_t const& handle,
  vertex_t const* keys /* map_key_first <= keys[] < map_key_last */,
  value_t const* values,
  size_t num_pairs,
  vertex_t const* renumber_map_labels,
  vertex_t map_key_first,
  vertex_t map_key_last);

template <typename vertex_t, typename value_t>
rmm::device_uvector<value_t> sort_values_by_key(raft::handle_t const& handle,
                                                vertex_t const* keys,
                                                value_t const* values,
                                                size_t num_pairs);

}  // namespace test
}  // namespace cugraph

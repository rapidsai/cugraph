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

#pragma once

#include <cugraph/graph.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace test {

template <typename vertex_t>
bool compare_renumbered_vectors(raft::handle_t const& handle,
                                std::vector<vertex_t> const& v1,
                                std::vector<vertex_t> const& v2);

template <typename vertex_t>
bool compare_renumbered_vectors(raft::handle_t const& handle,
                                rmm::device_uvector<vertex_t> const& v1,
                                rmm::device_uvector<vertex_t> const& v2);

template <typename T>
void single_gpu_renumber_edgelist_given_number_map(
  raft::handle_t const& handle,
  rmm::device_uvector<T>& d_edgelist_srcs,
  rmm::device_uvector<T>& d_edgelist_dsts,
  rmm::device_uvector<T>& d_renumber_map_gathered_v);

}  // namespace test
}  // namespace cugraph

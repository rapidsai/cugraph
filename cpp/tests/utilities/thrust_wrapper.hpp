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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace test {

template <typename key_buffer_type, typename value_buffer_type>
std::tuple<key_buffer_type, value_buffer_type> sort_by_key(raft::handle_t const& handle,
                                                           key_buffer_type const& keys,
                                                           value_buffer_type const& values);

template <typename vertex_t>
vertex_t max_element(raft::handle_t const& handle, raft::device_span<vertex_t const> vertices);

template <typename vertex_t>
void translate_vertex_ids(raft::handle_t const& handle,
                          rmm::device_uvector<vertex_t>& vertices /* [INOUT] */,
                          vertex_t vertex_id_offset);

template <typename vertex_t>
void populate_vertex_ids(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& d_vertices_v /* [INOUT] */,
                         vertex_t vertex_id_offset);

template <typename T>
rmm::device_uvector<T> randomly_select(raft::handle_t const& handle,
                                       rmm::device_uvector<T> const& input,
                                       size_t count,
                                       bool sort_results = false);

template <typename vertex_t, typename weight_t>
void remove_self_loops(raft::handle_t const& handle,
                       rmm::device_uvector<vertex_t>& d_src_v /* [INOUT] */,
                       rmm::device_uvector<vertex_t>& d_dst_v /* [INOUT] */,
                       std::optional<rmm::device_uvector<weight_t>>& d_weight_v /* [INOUT] */);

template <typename vertex_t, typename weight_t>
void sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<vertex_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<weight_t>>& d_weight_v /* [INOUT] */);

}  // namespace test
}  // namespace cugraph

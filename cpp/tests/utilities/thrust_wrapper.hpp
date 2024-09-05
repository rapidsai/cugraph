/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace test {

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> sort(
  raft::handle_t const& handle, cugraph::dataframe_buffer_type_t<value_t> const& values);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> sort(raft::handle_t const& handle,
                                               cugraph::dataframe_buffer_type_t<value_t>&& values);

template <typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<value_t>, cugraph::dataframe_buffer_type_t<value_t>>
sort(raft::handle_t const& handle,
     cugraph::dataframe_buffer_type_t<value_t> const& first,
     cugraph::dataframe_buffer_type_t<value_t> const& second);

template <typename key_t, typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<key_t>, cugraph::dataframe_buffer_type_t<value_t>>
sort_by_key(raft::handle_t const& handle,
            cugraph::dataframe_buffer_type_t<key_t> const& keys,
            cugraph::dataframe_buffer_type_t<value_t> const& values);

template <typename key_t, typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<key_t>, cugraph::dataframe_buffer_type_t<value_t>>
sort_by_key(raft::handle_t const& handle,
            cugraph::dataframe_buffer_type_t<key_t>&& keys,
            cugraph::dataframe_buffer_type_t<value_t>&& values);

template <typename key_t, typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<key_t>,
           cugraph::dataframe_buffer_type_t<key_t>,
           cugraph::dataframe_buffer_type_t<value_t>>
sort_by_key(raft::handle_t const& handle,
            cugraph::dataframe_buffer_type_t<key_t> const& first,
            cugraph::dataframe_buffer_type_t<key_t> const& second,
            cugraph::dataframe_buffer_type_t<value_t> const& values);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> unique(
  raft::handle_t const& handle, cugraph::dataframe_buffer_type_t<value_t>&& values);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> sequence(raft::handle_t const& handle,
                                                   size_t length,
                                                   size_t repeat_count,
                                                   value_t init);

// return (init + i) % modulo, where i = [0, length)
template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> modulo_sequence(raft::handle_t const& handle,
                                                          size_t length,
                                                          value_t modulo,
                                                          value_t init);

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

template <typename idx_t, typename offset_t>
void expand_sparse_offsets(raft::handle_t const& handle,
                           raft::device_span<offset_t const> offsets,
                           raft::device_span<idx_t> indices,
                           offset_t base_offset,
                           idx_t base_idx);

template <typename idx_t, typename offset_t>
void expand_hypersparse_offsets(raft::handle_t const& handle,
                                raft::device_span<offset_t const> offsets,
                                raft::device_span<idx_t const> nzd_indices,
                                raft::device_span<idx_t> indices,
                                offset_t base_offset);

}  // namespace test
}  // namespace cugraph

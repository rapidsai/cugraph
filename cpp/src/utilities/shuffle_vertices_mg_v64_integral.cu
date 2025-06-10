/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "shuffle_vertices.cuh"

namespace cugraph {

namespace detail {

template rmm::device_uvector<int64_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<int32_t>&& d_values,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<int64_t>&& d_values,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<size_t>&& d_values,
  raft::host_span<int64_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace detail

template rmm::device_uvector<int64_t> shuffle_ext_vertices(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t>&& vertices,
                               rmm::device_uvector<int32_t>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t>&& vertices,
                               rmm::device_uvector<int64_t>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t>&& vertices,
                               rmm::device_uvector<size_t>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace cugraph

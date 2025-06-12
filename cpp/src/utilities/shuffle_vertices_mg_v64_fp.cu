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

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t>&& vertices,
                               rmm::device_uvector<float>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t>&& vertices,
                               rmm::device_uvector<double>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_keys_with_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& keys,
  std::vector<arithmetic_device_uvector_t>&& properties,
  cugraph::detail::compute_gpu_id_from_int_vertex_t<int64_t> key_to_gpu_op);

template std::tuple<rmm::device_uvector<int64_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_keys_with_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& keys,
  std::vector<arithmetic_device_uvector_t>&& properties,
  cugraph::detail::compute_gpu_id_from_ext_vertex_t<int64_t> key_to_gpu_op);

template std::tuple<rmm::device_uvector<int64_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_keys_with_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& keys,
  std::vector<arithmetic_device_uvector_t>&& properties,
  cugraph::detail::compute_gpu_id_from_ext_edge_id_t<int64_t> key_to_gpu_op);

}  // namespace cugraph

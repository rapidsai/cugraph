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
#include <detail/shuffle_wrappers_impl.cuh>

namespace cugraph {
namespace detail {

template rmm::device_uvector<int32_t> shuffle_int_vertices_by_gpu_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  std::vector<int32_t> const& vertex_partition_range_lasts);
template rmm::device_uvector<int64_t> shuffle_int_vertices_by_gpu_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  std::vector<int64_t> const& vertex_partition_range_lasts);

template rmm::device_uvector<int32_t> shuffle_ext_vertices_by_gpu_id(
  raft::handle_t const& handle, rmm::device_uvector<int32_t>&& d_vertices);

template rmm::device_uvector<int64_t> shuffle_ext_vertices_by_gpu_id(
  raft::handle_t const& handle, rmm::device_uvector<int64_t>&& d_vertices);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
shuffle_ext_vertices_and_values_by_gpu_id(raft::handle_t const& handle,
                                          rmm::device_uvector<int32_t>&& d_vertices,
                                          rmm::device_uvector<int32_t>&& d_values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
shuffle_ext_vertices_and_values_by_gpu_id(raft::handle_t const& handle,
                                          rmm::device_uvector<int32_t>&& d_vertices,
                                          rmm::device_uvector<float>&& d_values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
shuffle_ext_vertices_and_values_by_gpu_id(raft::handle_t const& handle,
                                          rmm::device_uvector<int32_t>&& d_vertices,
                                          rmm::device_uvector<double>&& d_values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
shuffle_ext_vertices_and_values_by_gpu_id(raft::handle_t const& handle,
                                          rmm::device_uvector<int64_t>&& d_vertices,
                                          rmm::device_uvector<int64_t>&& d_values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
shuffle_ext_vertices_and_values_by_gpu_id(raft::handle_t const& handle,
                                          rmm::device_uvector<int64_t>&& d_vertices,
                                          rmm::device_uvector<float>&& d_values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
shuffle_ext_vertices_and_values_by_gpu_id(raft::handle_t const& handle,
                                          rmm::device_uvector<int64_t>&& d_vertices,
                                          rmm::device_uvector<double>&& d_values);

}  // namespace detail
}  // namespace cugraph

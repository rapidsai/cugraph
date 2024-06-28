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

#include "detail/graph_partition_utils.cuh"
#include "shuffle_vertices.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace detail {

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<float>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<double>&& values);

}  // namespace detail

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
shuffle_external_vertex_value_pairs(raft::handle_t const& handle,
                                    rmm::device_uvector<int64_t>&& vertices,
                                    rmm::device_uvector<float>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
shuffle_external_vertex_value_pairs(raft::handle_t const& handle,
                                    rmm::device_uvector<int64_t>&& vertices,
                                    rmm::device_uvector<double>&& values);

}  // namespace cugraph

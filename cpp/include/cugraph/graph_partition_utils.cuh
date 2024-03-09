/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cugraph/partition_manager.hpp>

#include <raft/core/device_span.hpp>

#include <cuco/hash_functions.cuh>

namespace cugraph {

template <typename vertex_t>
struct compute_gpu_id_from_ext_vertex_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(vertex_t v) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto vertex_partition_id = static_cast<int>(hash_func(v) % comm_size);
    return partition_manager::compute_global_comm_rank_from_vertex_partition_id(
      major_comm_size, minor_comm_size, vertex_partition_id);
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_ext_edge_endpoints_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_vertex_partition_id = static_cast<int>(hash_func(major) % comm_size);
    auto minor_vertex_partition_id = static_cast<int>(hash_func(minor) % comm_size);
    auto major_comm_rank           = major_vertex_partition_id % major_comm_size;
    auto minor_comm_rank           = minor_vertex_partition_id / major_comm_size;
    return partition_manager::compute_global_comm_rank_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
  }

  __host__ __device__ int operator()(
    thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_vertex_partition_id = static_cast<int>(hash_func(thrust::get<0>(pair)) % comm_size);
    auto minor_vertex_partition_id = static_cast<int>(hash_func(thrust::get<1>(pair)) % comm_size);
    auto major_comm_rank           = major_vertex_partition_id % major_comm_size;
    auto minor_comm_rank           = minor_vertex_partition_id / major_comm_size;
    return partition_manager::compute_global_comm_rank_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
  }
};

}  // namespace cugraph

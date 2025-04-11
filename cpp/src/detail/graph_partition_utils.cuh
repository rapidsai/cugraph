/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuco/hash_functions.cuh>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct compute_gpu_id_from_ext_vertex_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(vertex_t v) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
    auto vertex_partition_id = static_cast<int>(hash_func(v) % comm_size);
    return partition_manager::compute_global_comm_rank_from_vertex_partition_id(
      major_comm_size, minor_comm_size, vertex_partition_id);
  }
};

template <typename edge_t>
struct compute_gpu_id_from_ext_edge_id_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(edge_t e) const
  {
    cuco::murmurhash3_32<edge_t> hash_func{};
    auto vertex_partition_id = static_cast<int>(hash_func(e) % comm_size);
    return partition_manager::compute_global_comm_rank_from_vertex_partition_id(
      major_comm_size, minor_comm_size, vertex_partition_id);
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_int_vertex_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  int major_comm_size{0};
  int minor_comm_size{0};

  __device__ int operator()(vertex_t v) const
  {
    auto vertex_partition_id = static_cast<int>(cuda::std::distance(
      vertex_partition_range_lasts.begin(),
      thrust::upper_bound(
        thrust::seq, vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.end(), v)));
    return partition_manager::compute_global_comm_rank_from_vertex_partition_id(
      major_comm_size, minor_comm_size, vertex_partition_id);
  }
};

template <typename vertex_t>
struct compute_vertex_partition_id_from_ext_vertex_t {
  int comm_size{0};

  __host__ __device__ int operator()(vertex_t v) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
    return hash_func(v) % comm_size;
  }
};

template <typename vertex_t>
struct compute_vertex_partition_id_from_int_vertex_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};

  __device__ int operator()(vertex_t v) const
  {
    return static_cast<int>(cuda::std::distance(
      vertex_partition_range_lasts.begin(),
      thrust::upper_bound(
        thrust::seq, vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.end(), v)));
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_ext_edge_endpoints_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
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
    cuco::murmurhash3_32<vertex_t> hash_func{};
    auto major_vertex_partition_id = static_cast<int>(hash_func(thrust::get<0>(pair)) % comm_size);
    auto minor_vertex_partition_id = static_cast<int>(hash_func(thrust::get<1>(pair)) % comm_size);
    auto major_comm_rank           = major_vertex_partition_id % major_comm_size;
    auto minor_comm_rank           = minor_vertex_partition_id / major_comm_size;
    return partition_manager::compute_global_comm_rank_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_int_edge_endpoints_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    auto major_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               major)));
    auto minor_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               minor)));
    auto major_comm_rank = major_vertex_partition_id % major_comm_size;
    auto minor_comm_rank = minor_vertex_partition_id / major_comm_size;
    return partition_manager::compute_global_comm_rank_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
  }

  __device__ int operator()(thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    auto major_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               thrust::get<0>(pair))));
    auto minor_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               thrust::get<1>(pair))));
    auto major_comm_rank = major_vertex_partition_id % major_comm_size;
    auto minor_comm_rank = minor_vertex_partition_id / major_comm_size;
    return partition_manager::compute_global_comm_rank_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
  }
};

template <typename vertex_t>
struct compute_edge_partition_id_from_ext_edge_endpoints_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
    return (hash_func(major) % comm_size) * minor_comm_size +
           (hash_func(minor) % comm_size) / major_comm_size;
  }

  __host__ __device__ int operator()(
    thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
    return (hash_func(thrust::get<0>(pair)) % comm_size) * minor_comm_size +
           (hash_func(thrust::get<1>(pair)) % comm_size) / major_comm_size;
  }
};

template <typename vertex_t>
struct compute_edge_partition_id_from_int_edge_endpoints_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  int major_comm_size{0};
  int minor_comm_size{0};

  __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    auto major_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               major)));
    auto minor_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               minor)));
    return (major_vertex_partition_id)*minor_comm_size +
           (minor_vertex_partition_id) / major_comm_size;
  }

  __device__ int operator()(thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    auto major_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               thrust::get<0>(pair))));
    auto minor_vertex_partition_id =
      static_cast<int>(cuda::std::distance(vertex_partition_range_lasts.begin(),
                                           thrust::upper_bound(thrust::seq,
                                                               vertex_partition_range_lasts.begin(),
                                                               vertex_partition_range_lasts.end(),
                                                               thrust::get<1>(pair))));
    return (major_vertex_partition_id)*minor_comm_size +
           (minor_vertex_partition_id) / major_comm_size;
  }
};

// assumes that the edges are local to this GPU
template <typename vertex_t>
struct compute_local_edge_partition_id_from_ext_edge_endpoints_t {
  int comm_size{0};
  int major_comm_size{0};
  int minor_comm_size{0};

  __host__ __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    return compute_edge_partition_id_from_ext_edge_endpoints_t<vertex_t>{
             comm_size, major_comm_size, minor_comm_size}(major, minor) /
           comm_size;
  }

  __host__ __device__ int operator()(
    thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    return compute_edge_partition_id_from_ext_edge_endpoints_t<vertex_t>{
             comm_size, major_comm_size, minor_comm_size}(thrust::get<0>(pair),
                                                          thrust::get<1>(pair)) /
           comm_size;
  }
};

// assumes that the edges are local to this GPU
template <typename vertex_t>
struct compute_local_edge_partition_id_from_int_edge_endpoints_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  int major_comm_size{0};
  int minor_comm_size{0};

  __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    return compute_edge_partition_id_from_int_edge_endpoints_t<vertex_t>{
             vertex_partition_range_lasts, major_comm_size, minor_comm_size}(major, minor) /
           static_cast<int>(vertex_partition_range_lasts.size());
  }

  __device__ int operator()(thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    return compute_edge_partition_id_from_int_edge_endpoints_t<vertex_t>{
             vertex_partition_range_lasts, major_comm_size, minor_comm_size}(thrust::get<0>(pair),
                                                                             thrust::get<1>(pair)) /
           static_cast<int>(vertex_partition_range_lasts.size());
  }
};

struct compute_local_edge_partition_major_range_vertex_partition_id_t {
  int major_comm_size{};
  int minor_comm_size{};
  int major_comm_rank{};
  int minor_comm_rank{};

  __host__ __device__ int operator()(size_t local_edge_partition_idx) const
  {
    return partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, local_edge_partition_idx);
  }
};

struct compute_local_edge_partition_minor_range_vertex_partition_id_t {
  int major_comm_size{};
  int minor_comm_size{};
  int major_comm_rank{};
  int minor_comm_rank{};

  __host__ __device__ int operator()(size_t intra_partition_segment_idx) const
  {
    return partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, intra_partition_segment_idx, minor_comm_rank);
  }
};

}  // namespace detail
}  // namespace cugraph

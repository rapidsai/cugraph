/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/detail/hash_functions.cuh>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct compute_gpu_id_from_ext_vertex_t {
  int comm_size{0};

  __host__ __device__ int operator()(vertex_t v) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    return hash_func(v) % comm_size;
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_int_vertex_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};

  __host__ __device__ int operator()(vertex_t v) const
  {
    return static_cast<int>(thrust::distance(
      vertex_partition_range_lasts.begin(),
      thrust::upper_bound(
        thrust::seq, vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.end(), v)));
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_ext_edge_endpoints_t {
  int comm_size{0};
  int row_comm_size{0};
  int col_comm_size{0};

  __host__ __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_comm_rank = static_cast<int>(hash_func(major) % comm_size);
    auto minor_comm_rank = static_cast<int>(hash_func(minor) % comm_size);
    return (minor_comm_rank / row_comm_size) * row_comm_size + (major_comm_rank % row_comm_size);
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_int_edge_endpoints_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  int comm_size{0};
  int row_comm_size{0};
  int col_comm_size{0};

  __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    auto major_comm_rank =
      static_cast<int>(thrust::distance(vertex_partition_range_lasts.begin(),
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_partition_range_lasts.begin(),
                                                            vertex_partition_range_lasts.end(),
                                                            major)));
    auto minor_comm_rank =
      static_cast<int>(thrust::distance(vertex_partition_range_lasts.begin(),
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_partition_range_lasts.begin(),
                                                            vertex_partition_range_lasts.end(),
                                                            minor)));
    return (minor_comm_rank / row_comm_size) * row_comm_size + (major_comm_rank % row_comm_size);
  }

  __device__ int operator()(thrust::tuple<vertex_t, vertex_t> pair /* major, minor */) const
  {
    auto major_comm_rank =
      static_cast<int>(thrust::distance(vertex_partition_range_lasts.begin(),
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_partition_range_lasts.begin(),
                                                            vertex_partition_range_lasts.end(),
                                                            thrust::get<0>(pair))));
    auto minor_comm_rank =
      static_cast<int>(thrust::distance(vertex_partition_range_lasts.begin(),
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_partition_range_lasts.begin(),
                                                            vertex_partition_range_lasts.end(),
                                                            thrust::get<1>(pair))));
    return (minor_comm_rank / row_comm_size) * row_comm_size + (major_comm_rank % row_comm_size);
  }
};

template <typename vertex_t>
struct compute_partition_id_from_ext_edge_endpoints_t {
  int comm_size{0};
  int row_comm_size{0};
  int col_comm_size{0};

  __host__ __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_comm_rank = static_cast<int>(hash_func(major) % comm_size);
    auto minor_comm_rank = static_cast<int>(hash_func(minor) % comm_size);
    return major_comm_rank * col_comm_size + minor_comm_rank / row_comm_size;
  }
};

}  // namespace detail
}  // namespace cugraph

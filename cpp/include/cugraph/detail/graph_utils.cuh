/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <cuco/detail/hash_functions.cuh>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct compute_gpu_id_from_vertex_t {
  int comm_size{0};

  __device__ int operator()(vertex_t v) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    return hash_func(v) % comm_size;
  }
};

template <typename vertex_t>
struct compute_gpu_id_from_edge_t {
  int comm_size{0};
  int row_comm_size{0};
  int col_comm_size{0};

  __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_comm_rank = static_cast<int>(hash_func(major) % comm_size);
    auto minor_comm_rank = static_cast<int>(hash_func(minor) % comm_size);
    return (minor_comm_rank / row_comm_size) * row_comm_size + (major_comm_rank % row_comm_size);
  }
};

template <typename vertex_t>
struct compute_partition_id_from_edge_t {
  int comm_size{0};
  int row_comm_size{0};
  int col_comm_size{0};

  __device__ int operator()(vertex_t major, vertex_t minor) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_comm_rank = static_cast<int>(hash_func(major) % comm_size);
    auto minor_comm_rank = static_cast<int>(hash_func(minor) % comm_size);
    return major_comm_rank * col_comm_size + minor_comm_rank / row_comm_size;
  }
};

}  // namespace detail
}  // namespace cugraph

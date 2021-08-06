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

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <cuco/detail/hash_functions.cuh>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {
namespace detail {

// compute the numbers of nonzeros in rows (of the graph adjacency matrix, if store_transposed =
// false) or columns (of the graph adjacency matrix, if store_transposed = true)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degrees(
  raft::handle_t const& handle,
  std::vector<edge_t const*> const& adj_matrix_partition_offsets,
  std::optional<std::vector<vertex_t const*>> const& adj_matrix_partition_dcs_nzd_vertices,
  std::optional<std::vector<vertex_t>> const& adj_matrix_partition_dcs_nzd_vertex_counts,
  partition_t<vertex_t> const& partition,
  std::optional<std::vector<vertex_t>> const& adj_matrix_partition_segment_offsets)
{
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();

  auto use_dcs = adj_matrix_partition_dcs_nzd_vertices.has_value();

  rmm::device_uvector<edge_t> local_degrees(0, handle.get_stream());
  rmm::device_uvector<edge_t> degrees(0, handle.get_stream());

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < col_comm_size; ++i) {
    auto vertex_partition_idx  = static_cast<size_t>(i * row_comm_size + row_comm_rank);
    auto vertex_partition_size = partition.get_vertex_partition_size(vertex_partition_idx);
    max_num_local_degrees      = std::max(max_num_local_degrees, vertex_partition_size);
    if (i == col_comm_rank) { degrees.resize(vertex_partition_size, handle.get_stream()); }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < col_comm_size; ++i) {
    auto vertex_partition_idx = static_cast<size_t>(i * row_comm_size + row_comm_rank);
    vertex_t major_first{};
    vertex_t major_last{};
    std::tie(major_first, major_last) = partition.get_vertex_partition_range(vertex_partition_idx);
    auto p_offsets                    = adj_matrix_partition_offsets[i];
    auto major_hypersparse_first =
      use_dcs ? major_first + (*adj_matrix_partition_segment_offsets)
                                [(detail::num_sparse_segments_per_vertex_partition + 2) * i +
                                 detail::num_sparse_segments_per_vertex_partition]
              : major_last;
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(major_hypersparse_first - major_first),
                      local_degrees.begin(),
                      [p_offsets] __device__(auto i) { return p_offsets[i + 1] - p_offsets[i]; });
    if (use_dcs) {
      auto p_dcs_nzd_vertices   = (*adj_matrix_partition_dcs_nzd_vertices)[i];
      auto dcs_nzd_vertex_count = (*adj_matrix_partition_dcs_nzd_vertex_counts)[i];
      thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   local_degrees.begin() + (major_hypersparse_first - major_first),
                   local_degrees.begin() + (major_last - major_first),
                   edge_t{0});
      thrust::for_each(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       thrust::make_counting_iterator(vertex_t{0}),
                       thrust::make_counting_iterator(dcs_nzd_vertex_count),
                       [p_offsets,
                        p_dcs_nzd_vertices,
                        major_first,
                        major_hypersparse_first,
                        local_degrees = local_degrees.data()] __device__(auto i) {
                         auto d = p_offsets[(major_hypersparse_first - major_first) + i + 1] -
                                  p_offsets[(major_hypersparse_first - major_first) + i];
                         auto v                         = p_dcs_nzd_vertices[i];
                         local_degrees[v - major_first] = d;
                       });
    }
    col_comm.reduce(local_degrees.data(),
                    i == col_comm_rank ? degrees.data() : static_cast<edge_t*>(nullptr),
                    static_cast<size_t>(major_last - major_first),
                    raft::comms::op_t::SUM,
                    i,
                    handle.get_stream());
  }

  return degrees;
}

// compute the numbers of nonzeros in rows (of the graph adjacency matrix, if store_transposed =
// false) or columns (of the graph adjacency matrix, if store_transposed = true)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degrees(raft::handle_t const& handle,
                                                  edge_t const* offsets,
                                                  vertex_t number_of_vertices)
{
  rmm::device_uvector<edge_t> degrees(number_of_vertices, handle.get_stream());
  thrust::tabulate(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   degrees.begin(),
                   degrees.end(),
                   [offsets] __device__(auto i) { return offsets[i + 1] - offsets[i]; });
  return degrees;
}

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

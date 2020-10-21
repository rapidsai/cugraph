/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <experimental/graph_view.hpp>
#include <partition_manager.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <vector>

namespace cugraph {
namespace experimental {
namespace detail {

// compute the numbers of nonzeros in rows (of the graph adjacency matrix, if store_transposed =
// false) or columns (of the graph adjacency matrix, if store_transposed = true)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degree(
  raft::handle_t const &handle,
  std::vector<edge_t const *> const &adj_matrix_partition_offsets,
  partition_t<vertex_t> const &partition)
{
  auto &row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto &col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();

  rmm::device_uvector<edge_t> local_degrees(0, handle.get_stream());
  rmm::device_uvector<edge_t> degrees(0, handle.get_stream());

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < (partition.is_hypergraph_partitioned() ? col_comm_size : row_comm_size);
       ++i) {
    auto vertex_partition_idx = partition.is_hypergraph_partitioned()
                                  ? static_cast<size_t>(i * row_comm_size + row_comm_rank)
                                  : static_cast<size_t>(col_comm_rank * row_comm_size + i);
    auto vertex_partition_size = partition.get_vertex_partition_size(vertex_partition_idx);
    max_num_local_degrees      = std::max(max_num_local_degrees, vertex_partition_size);
    if (i == (partition.is_hypergraph_partitioned() ? col_comm_rank : row_comm_rank)) {
      degrees.resize(vertex_partition_size, handle.get_stream());
    }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < (partition.is_hypergraph_partitioned() ? col_comm_size : row_comm_size);
       ++i) {
    auto vertex_partition_idx = partition.is_hypergraph_partitioned()
                                  ? static_cast<size_t>(i * row_comm_size + row_comm_rank)
                                  : static_cast<size_t>(col_comm_rank * row_comm_size + i);
    vertex_t major_first{};
    vertex_t major_last{};
    std::tie(major_first, major_last) = partition.get_vertex_partition_range(vertex_partition_idx);
    auto p_offsets =
      partition.is_hypergraph_partitioned()
        ? adj_matrix_partition_offsets[i]
        : adj_matrix_partition_offsets[0] +
            (major_first - partition.get_vertex_partition_first(col_comm_rank * row_comm_size));
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(major_last - major_first),
                      local_degrees.data(),
                      [p_offsets] __device__(auto i) { return p_offsets[i + 1] - p_offsets[i]; });
    if (partition.is_hypergraph_partitioned()) {
      col_comm.reduce(local_degrees.data(),
                      i == col_comm_rank ? degrees.data() : static_cast<edge_t *>(nullptr),
                      static_cast<size_t>(major_last - major_first),
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
    } else {
      row_comm.reduce(local_degrees.data(),
                      i == row_comm_rank ? degrees.data() : static_cast<edge_t *>(nullptr),
                      static_cast<size_t>(major_last - major_first),
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
    }
  }

  raft::comms::status_t status{};
  if (partition.is_hypergraph_partitioned()) {
    status =
      col_comm.sync_stream(handle.get_stream());  // this is neessary as local_degrees will become
                                                  // out-of-scope once this function returns.
  } else {
    status =
      row_comm.sync_stream(handle.get_stream());  // this is neessary as local_degrees will become
                                                  // out-of-scope once this function returns.
  }
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  return degrees;
}

// compute the numbers of nonzeros in rows (of the graph adjacency matrix, if store_transposed =
// false) or columns (of the graph adjacency matrix, if store_transposed = true)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degree(
  raft::handle_t const &handle,
  std::vector<rmm::device_uvector<edge_t>> const &adj_matrix_partition_offsets,
  partition_t<vertex_t> const &partition)
{
  // we can avoid creating this temporary with "if constexpr" supported from C++17
  std::vector<edge_t const *> tmp_offsets(adj_matrix_partition_offsets.size(), nullptr);
  std::transform(adj_matrix_partition_offsets.begin(),
                 adj_matrix_partition_offsets.end(),
                 tmp_offsets.begin(),
                 [](auto const &offsets) { return offsets.data(); });
  return compute_major_degree(handle, tmp_offsets, partition);
}

template <typename vertex_t, typename edge_t>
struct degree_from_offsets_t {
  edge_t const *offsets{nullptr};

  __device__ edge_t operator()(vertex_t v) { return offsets[v + 1] - offsets[v]; }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

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

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/transform.h>

#include <vector>

namespace cugraph {
namespace experimental {
namespace detail {

// compute the numbers of nonzeros in rows of the (transposed) graph adjacency matrix
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_row_degree(
  raft::handle_t const &handle,
  std::vector<rmm::device_uvector<edge_t>> const &adj_matrix_partition_offsets,
  partition_t<vertex_t> const &partition)
{
  auto &comm_p_row     = handle.get_subcomm(comm_p_row_key);
  auto comm_p_row_rank = comm_p_row.get_rank();
  auto comm_p_row_size = comm_p_row.get_size();
  auto &comm_p_col     = handle.get_subcomm(comm_p_col_key);
  auto comm_p_col_rank = comm_p_col.get_rank();
  auto comm_p_col_size = comm_p_col.get_size();

  rmm::device_uvector<edge_t> local_degrees(0, handle.get_stream());
  rmm::device_uvector<edge_t> degrees(0, handle.get_stream());

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < comm_p_col_size; ++i) {
    auto vertex_partition_id = partition.hypergraph_partitioned
                                 ? comm_p_row_size * i + comm_p_row_rank
                                 : comm_p_col_size * comm_p_row_rank + i;
    auto row_first        = partition.vertex_partition_offsets[vertex_partition_id];
    auto row_last         = partition.vertex_partition_offsets[vertex_partition_id + 1];
    max_num_local_degrees = std::max(max_num_local_degrees, row_last - row_first);
    if (i == comm_p_col_rank) { degrees.resize(row_last - row_first, handle.get_stream()); }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < comm_p_col_size; ++i) {
    auto vertex_partition_id = partition.hypergraph_partitioned
                                 ? comm_p_row_size * i + comm_p_row_rank
                                 : comm_p_col_size * comm_p_row_rank + i;
    auto row_first = partition.vertex_partition_offsets[vertex_partition_id];
    auto row_last  = partition.vertex_partition_offsets[vertex_partition_id + 1];
    auto p_offsets =
      partition.hypergraph_partitioned
        ? adj_matrix_partition_offsets[i].data()
        : adj_matrix_partition_offsets[0].data() +
            (row_first - partition.vertex_partition_offsets[comm_p_col_size * comm_p_row_rank]);
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(row_last - row_first),
                      local_degrees.data(),
                      [p_offsets] __device__(auto i) { return p_offsets[i + 1] - p_offsets[i]; });
    comm_p_row.reduce(local_degrees.data(),
                      i == comm_p_col_rank ? degrees.data() : static_cast<edge_t *>(nullptr),
                      degrees.size(),
                      raft::comms::op_t::SUM,
                      comm_p_col_rank,
                      handle.get_stream());
  }

  return degrees;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

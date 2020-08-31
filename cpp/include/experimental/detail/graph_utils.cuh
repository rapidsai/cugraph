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
    auto major_first         = partition.vertex_partition_offsets[vertex_partition_id];
    auto major_last          = partition.vertex_partition_offsets[vertex_partition_id + 1];
    max_num_local_degrees    = std::max(max_num_local_degrees, major_last - major_first);
    if (i == comm_p_col_rank) { degrees.resize(major_last - major_first, handle.get_stream()); }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < comm_p_col_size; ++i) {
    auto vertex_partition_id = partition.hypergraph_partitioned
                                 ? comm_p_row_size * i + comm_p_row_rank
                                 : comm_p_col_size * comm_p_row_rank + i;
    auto major_first         = partition.vertex_partition_offsets[vertex_partition_id];
    auto major_last          = partition.vertex_partition_offsets[vertex_partition_id + 1];
    auto p_offsets =
      partition.hypergraph_partitioned
        ? adj_matrix_partition_offsets[i]
        : adj_matrix_partition_offsets[0] +
            (major_first - partition.vertex_partition_offsets[comm_p_col_size * comm_p_row_rank]);
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(major_last - major_first),
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
  for (size_t i = 0; i < adj_matrix_partition_offsets.size(); ++i) {
    tmp_offsets[i] = adj_matrix_partition_offsets[i].data();
  }
  return compute_major_degree(handle, tmp_offsets, partition);
}

template <typename vertex_t>
void check_vertex_partition_offsets(std::vector<vertex_t> const &vertex_partition_offsets,
                                    vertex_t number_of_vertices)
{
  CUGRAPH_EXPECTS(
    std::is_sorted(vertex_partition_offsets.begin(), vertex_partition_offsets.end()),
    "Invalid API parameter: partition.vertex_partition_offsets values should be non-descending.");
  CUGRAPH_EXPECTS(vertex_partition_offsets[0] == vertex_t{0},
                  "Invalid API parameter: partition.vertex_partition_offsets[0] should be 0.");
  CUGRAPH_EXPECTS(vertex_partition_offsets.back() == number_of_vertices,
                  "Invalid API parameter: partition.vertex_partition_offsets.back() should be "
                  "number_of_vertices.");
}

template <typename vertex_t, typename edge_t>
struct degree_from_offsets_t {
  edge_t const *offsets{nullptr};

  __device__ edge_t operator()(vertex_t v) { return offsets[v + 1] - offsets[v]; }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

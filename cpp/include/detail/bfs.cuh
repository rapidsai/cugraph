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

#include <patterns.hpp>

#include <rmm/rmm.h>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>


namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphType, typename VertexIterator, typename ResultIterator, typename vertex_t>:w
void bfs_this_partition(
    raft::Handle handle, GraphType const& csr_graph,
    VertexIteraotr distance_first, VertexIteraotr predecessor_first,
    vertex_t starting_vertex,
    bool direction_optimizing = false, size_t depth_limit = std::numeric_limits<size_t>::max(),
    bool do_expensive_check = false) {
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>::value,
    "VertexIterator should point to a vertex_t value.");
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(is_csr<GraphType>::value, "GraphType should be CSR.");

  auto const num_vertices = csr_graph.get_number_of_vertices();
  vertex_t this_partition_vertex_first{};
  vertex_t this_partition_vertex_last{};
  std::tie(this_partition_vertex_first, this_partition_vertex_last) =
    csr_graph.get_this_partition_vertex_range();
  vertex_t this_partition_adj_matrix_row_vertex_first{};
  vertex_t this_partition_adj_matrix_row_vertex_last{};
  std::tie(this_partition_adj_matrix_row_vertex_first, this_partition_adj_matrix_row_vertex_last) =
    csr_graph.get_this_partition_adj_matrix_row_vertex_range();
  vertex_t this_partition_adj_matrix_col_vertex_first{};
  vertex_t this_partition_adj_matrix_col_vertex_last{};
  std::tie(this_partition_adj_matrix_col_vertex_first, this_partition_adj_matrix_col_vertex_last) =
    csr_graph.get_this_partition_adj_matrix_col_vertex_range();
  if (num_vertices == 0) {
    return;
  }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    csr_graph.is_directed(),
    "Invalid input argument: input graph should be directed.");
  CUGRAPH_EXPECTS(
    csr_graph.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");
  CUGRAPH_EXPECTS(
    (starting_vertex >= static_cast<vertex_t>(0)) && (starting_vertex < num_vertices),
    "Invalid input argument: starting vertex out-of-range.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. initialize distances and predecessors

  auto val_first =
    thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first));
  thrust::transform(
    thrust::make_counting_iterator(this_partition_vertex_first),
    thrust::make_counting_iterator(this_partition_vertex_last),
    val_first,
    [starting_vertex] __device__ (auto val) {
      auto distance = std::numeric_limits<vertex_t>::max();
      if (val == starting_vertex) {
        distance = static_cast<vertex_t>(0);
      }
      return thrust::make_tuple(distance, invalid_vertex_id<vertex_t>::value);
    });

  // 3. initialize BFS frontier

  enum class Bucket { cur, num_buckets };
  AdjMatrixRowFrontier row_vertex_froniter(csr_graph, Bucket::num_buckets);

  if ((starting_vertex >= this_partition_adj_matrix_row_vertex_first) &&
      (starting_vertex < this_partition_adj_matrix_row_vertex_last)) {
    adj_matrix_row_frontier.get_bucket(Bucket::cur).insert(starting_vertex);
  }

  // 4. BFS iteration

  size_t depth{0};
  auto cur_adj_matrix_row_frontier_first =
    adj_matrix_row_frontier.get_bucket(Bucket::cur).begin();
  auto cur_adj_matrix_row_frontier_aggregate_size =
    adj_matrix_row_frontier.get_bucket(Bucket::cur).aggregate_size();
  while (true) {
    if (direction_optimizing) {
      CUGRAPH_FAIL("unimplemented.");
    }
    else {
      auto cur_adj_matrix_row_frontier_last =
        adj_matrix_row_frontier.get_bucket(Bucket::cur).end();

      expand_and_transform_if_v_push_if_e(
        handle, csr_graph,
        cur_adj_matrix_row_frontier_first, cur_adj_matrix_row_frontier_last,
        thrust::make_counting_iterator(this_partition_adj_matrix_row_vertex_first),
        thrust::make_constant_iteraotr(this_partition_adj_matrix_col_vertex_first),
        distance_first,
        thrust::make_zip_iterator(distance_first, predecessor_first),
        row_frontier_queue,
        [distance_first, this_partition_vertex_first] __device__ (auto src_val, auto dst_val) {
          auto push = true;
          // FIXME: this check is unnecessary if not OPG, instead of taking opg as a template
          // parameter, it might be cleaner to take a graph device view object (similar to cuDF),
          // and implement check_local() which becomes a constexpr function always returning true
          // if not OPG.
          bool local =
            (dst_val >= this_partition_vertex_first) && (dst_val < this_partition_vertetx_last);
          if (local) {
            auto distance = *(distance_first + (dst_val - this_partition_vertex_first));
            if (distance != std::numeric_limits<vertex_t>::max()) {
              push = false;
            }
          }
          return thrust::make_tuple(push, src_val);
        },
        reduce_op::any<vertex_t>(),
        [] __device__ (auto v_val, auto pushed_val) {
          auto new_val = thrust::make_tuple(depth + 1, pushed_val);
          auto idx = AdjMatrixRowFrontier::invalid_bucket_idx;
          if (v_val == std::numeric_limits<vertex_t>::max()) {
            idx = Bucket::cur;
          }
          return thrust::make_tuple(idx, new_val);
        });

      auto new_adj_matrix_row_frontier_aggregate_size =
        adj_matrix_row_frontier.get_bucket(Bucket::cur).aggregate_size() -
        cur_adj_matrix_row_frontier_aggregate_size;
      if (new_adj_matrix_row_frontier_aggregate_size == 0) {
        break;
      }

      cur_adj_matrix_row_frontier_first = cur_adj_matrix_row_frontier_last;
      cur_adj_matrix_row_frontier_aggregate_size += new_frontier_aggregate_size;
    }

    depth++;
    if (depth >= depth_limit) {
      break;
    }
  }

  return;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

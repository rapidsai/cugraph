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
// FIXME: better move this file to include/utilities (following cuDF) and rename to error.hpp
#include <utilities/error_utils.h>

// FIXME: think about moving pattern accelerator API related files to detail/patterns
#include <algorithms.hpp>
#include <detail/graph_device_view.cuh>
#include <detail/patterns/adj_matrix_row_frontier.cuh>
#include <detail/patterns/expand_row_and_transform_if_e.cuh>
#include <detail/patterns/reduce_op.cuh>
#include <graph.hpp>

#include <raft/handle.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>


namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphType, typename VertexIterator>
void bfs_this_partition(
    raft::handle_t &handle, GraphType const &csr_graph,
    VertexIterator distance_first, VertexIterator predecessor_first,
    typename GraphType::vertex_type start_vertex, bool direction_optimizing, size_t depth_limit,
    bool do_expensive_check) {
  using vertex_t = typename GraphType::vertex_type;

  static_assert(
    std::is_integral<vertex_t>::value,
    "GraphType::vertex_type should be integral.");
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>::value,
    "GraphType::vertex_type and VertexIterator mismatch.");
  static_assert(GraphType::is_row_major, "GraphType should be CSR.");

  auto p_graph_device_view =
    graph_compressed_sparse_device_view_t<GraphType>::create(csr_graph);
  auto const graph_device_view = *p_graph_device_view;

  auto const num_vertices = graph_device_view.get_number_of_vertices();
  if (num_vertices == 0) {
    return;
  }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    graph_device_view.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");
  CUGRAPH_EXPECTS(
    graph_device_view.in_vertex_range(start_vertex),
    "Invalid input argument: starting vertex out-of-range.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<vertex_t>::max();
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  auto val_first =
    thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first));
  thrust::transform(
    thrust::cuda::par.on(handle.get_stream()),
    graph_device_view.this_partition_vertex_begin(),
    graph_device_view.this_partition_vertex_end(),
    val_first,
    [graph_device_view, start_vertex] __device__ (auto val) {
      auto distance = invalid_distance;
      auto v = graph_device_view.get_vertex_from_this_partition_vertex_offset_nocheck(val);
      if (v == start_vertex) {
        distance = static_cast<vertex_t>(0);
      }
      return thrust::make_tuple(distance, invalid_vertex);
    });

  // 3. initialize BFS frontier

  enum class Bucket { cur, num_buckets };
  std::vector<size_t> bucket_sizes(
    static_cast<size_t>(Bucket::num_buckets),
    graph_device_view.get_number_of_this_partition_adj_matrix_rows());
  AdjMatrixRowFrontier<
    raft::handle_t, thrust::tuple<vertex_t>, vertex_t, static_cast<size_t>(Bucket::num_buckets)
  > adj_matrix_row_frontier(handle, bucket_sizes);

  if (graph_device_view.in_this_partition_adj_matrix_row_range_nocheck(start_vertex)) {
    adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).insert(start_vertex);
  }

  // 4. BFS iteration

  vertex_t depth{0};
  auto cur_adj_matrix_row_frontier_first =
    adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).begin();
  auto cur_adj_matrix_row_frontier_aggregate_size =
    adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size();
  while (true) {
    if (direction_optimizing) {
      CUGRAPH_FAIL("unimplemented.");
    }
    else {
      auto cur_adj_matrix_row_frontier_last =
        adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).end();
      expand_row_and_transform_if_v_push_if_e(
        handle, graph_device_view,
        cur_adj_matrix_row_frontier_first, cur_adj_matrix_row_frontier_last,
        graph_device_view.this_partition_adj_matrix_row_begin(),
        graph_device_view.this_partition_adj_matrix_col_begin(),
        distance_first,
        thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first)),
        thrust::make_discard_iterator(), thrust::make_discard_iterator(),
        adj_matrix_row_frontier,
        [graph_device_view, distance_first] __device__ (auto row_val, auto col_val) {
          uint32_t push = true;
          bool local = graph_device_view.in_this_partition_vertex_range_nocheck(row_val);
          if (local) {
            auto this_partition_vertex_offset =
              graph_device_view.get_this_partition_vertex_offset_from_vertex_nocheck(col_val);
            auto distance = *(distance_first + this_partition_vertex_offset);
            if (distance != invalid_distance) {
              push = false;
            }
          }
          // FIXME: need to test this works properly if payload size is 0 (returns a tuple of size 1)
          return thrust::make_tuple(push, row_val);
        },
        reduce_op::any<thrust::tuple<vertex_t>>(),
        [depth] __device__ (auto v_val, auto pushed_val) {
          auto idx =
            AdjMatrixRowFrontier<
              raft::handle_t, thrust::tuple<vertex_t>, vertex_t
            >::kInvalidBucketIdx;
          if (v_val == invalid_distance) {
            idx = static_cast<size_t>(Bucket::cur);
          }
          return thrust::make_tuple(idx, depth + 1, thrust::get<0>(pushed_val));
        });

      auto new_adj_matrix_row_frontier_aggregate_size =
        adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() -
        cur_adj_matrix_row_frontier_aggregate_size;
      if (new_adj_matrix_row_frontier_aggregate_size == 0) {
        break;
      }

      cur_adj_matrix_row_frontier_first = cur_adj_matrix_row_frontier_last;
      cur_adj_matrix_row_frontier_aggregate_size += new_adj_matrix_row_frontier_aggregate_size;
    }

    depth++;
    if (depth >= static_cast<vertex_t>(depth_limit)) {
      break;
    }
  }

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
void bfs(raft::handle_t &handle, GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
         vertex_t *distances, vertex_t *predecessors, vertex_t start_vertex,
         bool direction_optimizing, size_t depth_limit, bool do_expensive_check) {
  detail::bfs_this_partition(
    handle, graph, distances, predecessors, start_vertex,
    direction_optimizing, depth_limit, do_expensive_check);
}

// explicit instantiation

template void bfs(
  raft::handle_t &handle, GraphCSRView<int32_t, int32_t, float> const &graph,
  int32_t *distances, int32_t *predecessors, int32_t start_vertex,
  bool direction_optimizing, size_t depth_limit, bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
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

template <typename GraphType, typename VertexIterator, typename ResultIterator, typename vertex_t>
void bfs_this_graph_partition(
    raft::Handle handle, GraphType const& graph,
    VertexIteraotr dst_distance_first, VertexIteraotr dst_predecessor_first,
    vertex_t starting_vertex,
    bool direction_optimized = false, size_t depth_limit = std::numeric_limits<size_t>::max()) {
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>::value,
    "VertexIterator should point to a vertex_t value.");
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(
    is_csr<GraphType>::value,
    "cugraph::experimental::bfs expects a CSR graph.");

  CUGRAPH_EXPECTS(
    graph.is_directed(), "cugraph::experimental::bfs expects a directed graph.");
  CUGRAPH_EXPECTS(
    graph.is_symmetric() || !direction_optimized,
    "cugraph::experimental::bfs expects a symmetric graph if direction optimize is true.");

  auto const num_vertices = graph.get_number_of_vertices();
  if (num_vertices == 0) {
    return;
  }

  vertex_t src_vertex_first{};
  vertex_t src_vertex_last{};
  vertex_t dst_vertex_first{};
  vertex_t dst_vertex_last{};
  std::tie(src_vertex_first, src_vertex_last) = graph.get_this_src_vertex_range();
  std::tie(dst_vertex_first, dst_vertex_last) = graph.get_this_dst_vertex_range();
  auto num_src_vertices = src_vertex_last - src_vertex_first;
  auto num_dst_vertices = dst_vertex_last - dst_vertex_first;

  auto dst_val_first =
    thrust::make_zip_iterator(thrust::make_tuple(dst_distance_first, dst_predecessor_first));
  thrust::transform(
    thrust::make_counting_iterator(dst_vertex_first),
    thrust::make_counting_iterator(dst_vertex_last),
    dst_val_first,
    [starting_vertex] __device__ (auto val) {
      auto distance = std::numeric_limits<vertex_t>::max();
      if (val == starting_vertex) {
        distance = static_cast<vertex_t>(0);
      }
      return thrust::make_tuple(distance, invalid_vertex_id<vertex_t>::value);
    });

  enum class Bucket { cur, num_buckets };
  SrcVertexQueue src_froniter_queue(graph, Bucket::num_buckets);

  if ((starting_vertex >= src_vertex_first) && (starting_vertex < src_vertex_last)) {
    src_frontier_queue.get_bucket(Bucket::cur).insert(starting_vertex);
  }
  if ((starting_vertex >= dst_vertex_first) && (starting_vertex < dst_vertex_last)) {
    *(dst_distance_first + (starting_vertex - dst_vertex_first)) = static_cast<vertex_t>(depth);
  }

  size_t depth{0};
  auto cur_src_frontier_first =
    src_frontier_queue.get_bucket(Bucket::cur).begin();
  while (true) {
    if (direction_optimized) {
      CUGRAPH_FAIL("unimplemented.");
    }
    else {
      auto cur_src_frontier_last =
        src_frontier_queue.get_bucket(Bucket::cur).end();

      for_each_src_v_expand_and_transform_if_e(
        handle, graph,
        cur_src_frontier_first, cur_src_frontier_last,
        thrust::make_counting_iterator(src_vertex_first), dst_distance_first,
        thrust::make_zip_iterator(dst_distance_first, dst_predecessor_first),
        src_frontier_queue,
        [] __device__ (auto src_val, auto dst_val) {
          int idx = !dst_val ? Bucket::cur : SrcVertexQueue::invalid_bucket_idx;
          return thrust::make_tuple(idx, thrust::make_tuple(depth + 1, src_val));
        });

      cur_src_frontier_first = cur_src_frontier_last;
      auto cur_src_frontier_size =
        static_cast<vertex_t>(
          thrust::distance(cur_src_froniter_first, src_frontier_queue.get_bucket(Bucket::cur).end())
        );

      auto aggregate_cur_src_frontier_size = handle.reduce(cur_src_frontier_size);
      if (aggregate_cur_src_frontier_size == 0) {
        break;
      }
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

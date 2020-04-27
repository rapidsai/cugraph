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
void sssp_this_graph_partition(
    raft::Handle handle, GraphType const& graph,
    VertexIteraotr src_distance_first, VertexIteraotr src_predecessor_first,
    vertex_t starting_vertex) {
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>::value,
    "VertexIterator should point to a vertex_t value.");
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(
    is_csr<GraphType>::value,
    "cugraph::experimental::sssp expects a CSR graph.");

  CUGRAPH_EXPECTS(
    graph.is_directed(), "cugraph::experimental::sssp expects a directed graph.");

  auto const num_vertices = graph.get_number_of_vertices();
  vertex_t src_vertex_first{};
  vertex_t src_vertex_last{};
  vertex_t dst_vertex_first{};
  vertex_t dst_vertex_last{};
  std::tie(src_vertex_first, src_vertex_last) = graph.get_this_src_vertex_range();
  std::tie(dst_vertex_first, dst_vertex_last) = graph.get_this_dst_vertex_range();
  auto num_src_vertices = src_vertex_last - src_vertex_first;
  auto num_dst_vertices = dst_vertex_last - dst_vertex_first;

  auto src_val_first =
    thrust::make_zip_iterator(thrust::make_tuple(src_distance_first, src_predecessor_first));
  thrust::transform(
    thrust::make_counting_iterator(src_vertex_first),
    thrust::make_counting_iterator(src_vertex_last),
    src_val_first,
    [starting_vertex] __device__ (auto val) {
      auto distance = std::numeric_limits<vertex_t>::max();
      if (val == starting_vertex) {
        distance = static_cast<weight_t>(0.0);
      }
      return thrust::make_tuple(distance, invalid_vertex_id<vertex_t>::value);
    });

  rmm::device_vector<vertex_t> cur_src_frontiers(num_src_vertices, invalid_vertex_id<vertex_t>::value);
  rmm::device_vector<vertex_t> new_src_frontiers(num_src_vertices, invalid_vertex_id<vertex_t>::value);
  rmm::device_vector<weight_t> dst_distances(
    num_dst_vertices, std::numeric_limits<weight_t>::max());

  vertex_t cur_src_frontier_size{0};
  if ((starting_vertex >= src_vertex_first) && (starting_vertex < src_vertex_last)) {
    cur_src_frontier_size = 1;
  }
  if ((starting_vertex >= dst_vertex_first) && (starting_vertex < dst_vertex_last)) {
    dst_distances[starting_vertex - dst_vertex_first] = static_cast<weight_t>(0.0);
  }
  while (true) {
    auto cur_src_frontier_first = cur_src_frontiers.begin();
    auto cur_src_frontier_last = cur_src_frontier_first + cur_src_frontier_size;

    copy_src_values_to_dst(
      handle, graph, cur_src_frontier_first, cur_src_frontier_last,
      thrust::make_transform_iterator(
        cur_src_frontier_first,
        [src_distance_first, src_vertex_first] __device__ (auto val) {
          return *(src_distance_first + (val - src_vertex_first));
        }),
      dst_distances.begin());

    // TODO: implement the near-far method to improve work efficiency

    auto new_src_froniter_last =
      for_each_src_v_expand_and_transform_if_e(
        handle, graph,
        cur_src_frontiers.begin(), cur_src_frontiers.begin() + cu_src_frontier_size,
        thrust::make_counting_iterator(src_vertex_first), dst_distances.begin(),
        thrust::make_zip_iterator(src_distance_first, src_predecessor_first),
        new_src_frontiers.begin(),
        [src_distance_first, src_vertex_first] __device__ (auto src_val, auto dst_val, weight_t w) {
          auto old_dist = dst_val;
          auto new_dist = *(src_distance_first + (src_val - src_vertex_first)) + w;
          return thrust::make_tuple(new_dist < old_dist, thrust::make_tuple(new_dist, src_val));
        });
        [] __device__ (auto val0, auto val1) {
          auto dist0 = thrust::get<0>(val0);
          auto dist1 = thrust::get<0>(val1);
          return dist0 <= dist1 ? val0 : val1;
        });
    cur_src_frontier_size =
      static_cast<vertex_t>(thrust::distance(new_src_frontiers.begin(), new_src_frontier_last));
    std::swap(cur_src_frontiers, new_src_frontiers);

    aggregate_cur_frontier_size = handle.reduce(cur_src_frontier_size);
    if (aggregate_cur_frontier_size == 0) {
      break;
    }
  }

  return;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

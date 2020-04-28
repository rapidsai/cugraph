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

  // implements the Near-Far method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.

  auto const num_vertices = graph.get_number_of_vertices();
  auto const num_edges = graph.get_number_of_edges();
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

  weight_t average_vertex_degree{0.0};
  weight_t average_edge_weight{0.0};
  std::tie(average_vertex_degree, average_edge_weight) =
    cugraph::transform_reduce_e(
      handle, graph,
      thrust::make_constant_iterator(0)/* dummy */, thrust::make_constant_iterator(0)/* dummy */,
      [] __device__ (auto src_val, auto dst_val, weight_t w) {
        return thrust::make_tuple(static_cast<weight_t>(1.0), w);
      },
      thrust::make_tuple(staitc_cast<weight_t>(0.0), static_cast<weight_t>(0.0)));
  average_vertex_degree /= static_cast<weight_t>(num_vertices);
  average_edge_weight /=
    num_edges > 0 ? static_cast<weight_t>(num_edges) : static_cast<weight_t>(1.0);
  auto delta = (warp_size * average_weight) / average_v_degree;

  auto dst_val_first =
    thrust::make_zip_iterator(thrust::make_tuple(dst_distance_first, dst_predecessor_first));
  thrust::transform(
    thrust::make_counting_iterator(dst_vertex_first),
    thrust::make_counting_iterator(dst_vertex_last),
    dst_val_first,
    [starting_vertex] __device__ (auto val) {
      auto distance = std::numeric_limits<vertex_t>::max();
      if (val == starting_vertex) {
        distance = static_cast<weight_t>(0.0);
      }
      return thrust::make_tuple(distance, invalid_vertex_id<vertex_t>::value);
    });

  enum class Bucket { cur_near, new_near, far, num_buckets };
  SrcVertexQueue src_frontier_queue(graph, Bucket::num_buckets);

  rmm::device_vector<weight_t> src_distances(
    num_src_vertices, std::numeric_limits<weight_t>::max());

  if ((starting_vertex >= src_vertex_first) && (starting_vertex < src_vertex_last)) {
    src_frontier_queue.get_bucket(Bucket::cur_near).insert(starting_vertex);
    src_distances[starting_vertex - src_vertex_first] = static_cast<weight_t>(0.0);
  }
  if ((starting_vertex >= dst_vertex_first) && (starting_vertex < dst_vertex_last)) {
    *(dst_distance_first + (starting_vertex - dst_vertex_first)) = static_cast<weight_t>(0.0);
  }
  auto near_far_threshold = delta;
  while (true) {
    for_each_src_v_expand_and_transform_if_e(
      handle, graph,
      src_frontier_queue.get_bucket(Bucket::cur_near).begin(),
      src_frontier_queue.get_bucket(Bucket::cur_near).end(),
      src_distances.begin(), dst_distance_first,
      thrust::make_zip_iterator(dst_distance_first, dst_predecessor_first),
      src_frontier_queue, src_distances.begin(),
      [near_far_threshold] __device__ (auto src_val, auto dst_val, weight_t w) {
        auto old_dist = dst_val;
        auto new_dist = src_val + w;
        int idx =
          new_dist < old_dist
          ? (new_dist < near_far_threshold ? Bucket::new_near : Bucket::far)
          : SrcVertexQueue::invalid_bucket_idx;
        return thrust::make_tuple(idx, thrust::make_tuple(new_dist, src_val));
      });
      [] __device__ (auto val0, auto val1) {
        auto dist0 = thrust::get<0>(val0);
        auto dist1 = thrust::get<0>(val1);
        return dist0 <= dist1 ? val0 : val1;
      },
      [] __device__ (auto val) {
        return thrust::get<0>(val);
      });
    src_frontier_queue.get_bucket(Bucket::cur_near).clear();
    if (src_frontier_queue.get_bucket(Bucket::new_near).size() > 0) {  // near queue is non-empty
      src_frontier_queue.swap_buckets(Bucket::cur_near, Bucket::new_near);
    }
    else {  // near queue is empty, split the far queue
      while (true) {
        near_far_threshold += delta;
        src_fontier_queue.get_bucket(Bucket::far).move_to(
          [near_far_threshold] __device__ (auto v) {
            auto dist = src_distances_first + (v - src_vertex_first);
            if (dist < near_far_threshold) {
              return Bucket::cur_near;
            }
            else {
              return Bucket::far;
            }
          });
        auto aggregate_cur_near_frontier_size =
          handle.reduce(src_froniter_queue.get_bucket(Bucket::cur_near).size());
        if (aggregate_cur_frontier_size > 0) {
          break;
        }
        else {
          auto aggregate_far_frontier_size =
            handle.reduce(src_froniter_queue.get_bucket(Bucket::far).size());
          if (aggregate_far_frontier_size == 0) {
            return;
          }
        }
      }
    }
  }

  return;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

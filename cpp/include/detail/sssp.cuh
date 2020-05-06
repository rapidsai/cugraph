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

template <typename GraphType, typename VertexIterator, typename WeightIterator, typename vertex_t,
          bool opg = false>
void sssp_this_partition(
    raft::Handle handle, GraphType const& csr_graph,
    WeightIteraotr distance_first, VertexIteraotr predecessor_first,
    vertex_t starting_vertex,
    size_t depth_limit = std::numeric_limits<size_t>::max(), bool do_expensive_check = false) {
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>::value,
    "VertexIterator should point to a vertex_t value.");
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(
    std::is_same<typename std::iterator_traits<WeightIterator>::value_type, weight_t>::value,
    "VertexIterator should point to a vertex_t value.");
  static_assert(is_csr<GraphType>::value, "GraphType should be CSR.");

  auto const num_vertices = csc_graph.get_number_of_vertices();
  auto const num_edges = csr_graph.get_number_of_edges();
  vertex_t this_partition_vertex_first{};
  vertex_t this_partition_vertex_last{};
  std::tie(this_partition_vertex_first, this_partition_vertex_last) =
    csc_graph.get_this_partition_vertex_range();
  auto this_partition_num_vertices =
    csr_graph.get_this_partition_number_of_vertices();
  vertex_t this_partition_adj_matrix_row_vertex_first{};
  vertex_t this_partition_adj_matrix_row_vertex_last{};
  std::tie(this_partition_adj_matrix_row_vertex_first, this_partition_adj_matrix_row_vertex_last) =
    csr_graph.get_this_partition_adj_matrix_row_vertex_range();
  auto adj_matrix_row_num_vertices =
    csr_graph.get_this_partition_adj_matrix_row_number_of_vertices();
  vertex_t this_partition_adj_matrix_col_vertex_first{};
  vertex_t this_partition_adj_matrix_col_vertex_last{};
  std::tie(this_partition_adj_matrix_col_vertex_first, this_partition_adj_matrix_col_vertex_last) =
    csr_graph.get_this_partition_adj_matrix_col_vertex_range();
  if (num_vertices == 0) {
    return;
  }

  // implements the Near-Far Pile method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    csc_graph.is_directed(),
    "Invalid input argument: input graph should be directed.");
  CUGRAPH_EXPECTS(
    (starting_vertex >= static_cast<vertex_t>(0)) && (starting_vertex < num_vertices),
    "Invalid input argument: starting vertex out-of-range.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. update delta

  weight_t average_vertex_degree{0.0};
  weight_t average_edge_weight{0.0};
  std::tie(average_vertex_degree, average_edge_weight) =
    transform_reduce_e(
      handle, csr_graph,
      thrust::make_constant_iterator(0)/* dummy */, thrust::make_constant_iterator(0)/* dummy */,
      [] __device__ (auto src_val, auto dst_val, weight_t w) {
        return thrust::make_tuple(static_cast<weight_t>(1.0), w);
      },
      thrust::make_tuple(staitc_cast<weight_t>(0.0), static_cast<weight_t>(0.0)));
  average_vertex_degree /= static_cast<weight_t>(num_vertices);
  average_edge_weight /=
    num_edges > 0 ? static_cast<weight_t>(num_edges) : static_cast<weight_t>(1.0);
  auto delta = (static_cast<weight_t>(handle.get_warp_size()) * average_weight) / average_v_degree;

  // 3. initialize distances and predecessors

  auto val_first =
    thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first));
  thrust::transform(
    thrust::make_counting_iterator(this_partition_vertex_first),
    thrust::make_counting_iterator(this_partition_vertex_last),
    val_first,
    [starting_vertex] __device__ (auto val) {
      auto distance = std::numeric_limits<vertex_t>::max();
      if (val == starting_vertex) {
        distance = static_cast<weight_t>(0.0);
      }
      return thrust::make_tuple(distance, invalid_vertex_id<vertex_t>::value);
    });

  // 4. initialize SSSP frontier

  enum class Bucket { cur_near, new_near, far, num_buckets };
  RowVertexFrontier row_vertex_frontier(csr_graph, Bucket::num_buckets);
  row_vertex_frontier.track_updated_vertices_in_this_partition();

  // 5. SSSP iteration

  rmm::device_vector<weight_t> adj_matrix_row_distances(
    num_adj_matrix_row_vertices, std::numeric_limits<weight_t>::max());

  if ((starting_vertex >= adj_matrix_row_vertex_first) &&
      (starting_vertex < adj_matrix_row_vertex_last)) {
    row_vertex_frontier.get_bucket(Bucket::cur_near).insert(starting_vertex);
    adj_matrix_row_distances[starting_vertex - adj_matrix_row_veretx_first] =
      static_cast<weight_t>(0.0);
  }

  auto near_far_threshold = delta;
  while (true) {
    row_vertex_frontier.clear_updated_vertices_in_this_partition();

    expand_and_update_if_v_push_if_e(
      handle, csr_graph,
      row_vertex_frontier.get_bucket(Bucket::cur_near).begin(),
      row_vertex_frontier.get_bucket(Bucket::cur_near).end(),
      thrust::make_zip_iterator(
        adj_matrix_row_distances.begin(),
        thrust::make_counting_iterator(this_partition_adj_matrix_row_vertex_first)),
      thrust::make_counting_iteraotr(this_partition_adj_matrix_col_vertex_first),
      distance_first,
      thrust::make_zip_iterator(distance_first, predecessor_first),
      row_frontier_queue,
      [distance_first, this_partition_vertex_first] __device__ (
        auto src_val, auto dst_val, weight_t w) {
        auto push = true;
        auto new_distance = thrust::get<0>(src_val) + w;
        bool local =
          opg
          ? (dst_val >= this_partition_vertex_first) && (dst_val < this_partition_vertetx_last)
          : true;
        if (local) {
          auto old_distance = *(distance_first + (dst_val - this_partition_vertex_first));
          if (new_distance >= old_distance) {
            push = false;
          }
        }
        return thrust::make_tuple(push, thrust::make_tuple(new_distance, thrust::get<1>(src_val)));
      },
      reduce_op::min_tuple<weight_t, vertex_t, 0>(),
      [near_far_threshold] __device__ (auto v_val, auto pushed_val) {
        auto new_dist = thrust::get<0>(pushed_val);
        auto idx =
          new_dist < v_val
          ? (new_dist < near_far_threshold ? Bucket::new_near : Bucket::far)
          : RowVertexFrontier::invalid_bucket_idx;
        return thrust::make_tuple(idx, pushed_val);
      });

    copy_to_adj_matrix_row(
      handle, csc_graph,
      row_vertex_frontier.get_updated_vertices_in_this_partition().begin(),
      row_vertex_frontier.get_updated_vertices_in_this_partition().end(),
      distance_first, adj_matrix_row_distances.begin());

    row_vertex_frontier.get_bucket(Bucket::cur_near).clear();
    if (row_vertex_frontier.get_bucket(Bucket::new_near).aggregate_size() > 0) {
      row_vertex_frontier.swap_buckets(Bucket::cur_near, Bucket::new_near);
    }
    else {  // near queue is empty, split the far queue
      auto old_near_far_threshold = near_far_threshold;
      near_far_threshold += delta;
      auto adj_matrix_row_distance_first = adj_matrix_row_distances.begin();
      while (true) {
        row_vertex_fontier.get_bucket(Bucket::far).split(
          [adj_matrix_row_distance_first, adj_matrix_row_vertex_first, old_near_far_threshold,
           near_far_threshold] __device__ (auto v) {
            auto dist = *(adj_matrix_row_distance_first + (v - adj_matrix_row_vertex_first));
            if (dist < old_near_far_threshold) {
              return RowVertexFrontier::invalid_bucket_idx;
            }
            else if (dist < near_far_threshold) {
              return Bucket::cur_near;
            }
            else {
              return Bucket::far;
            }
          });
        if (row_vertex_frontier.get_bucket(Bucket::cur_near).aggregate_size() > 0) {
          break;
        }
        else if (row_vertex_frontier.get_bucket(Bucket::far).aggregate_size() > 0) {
          near_far_threshold += delta;
        }
        else {
          return;
        }
      }
    }
  }

  return;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

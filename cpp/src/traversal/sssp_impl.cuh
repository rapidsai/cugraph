/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <prims/count_if_e.cuh>
#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_e.cuh>
#include <prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <prims/update_v_frontier.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/util/cudart_utils.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

namespace cugraph {

namespace {

template <typename vertex_t, typename weight_t, bool multi_gpu>
struct e_op_t {
  vertex_partition_device_view_t<vertex_t, multi_gpu> vertex_partition{};
  weight_t const* distances{};
  weight_t cutoff{};

  __device__ thrust::optional<thrust::tuple<weight_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, weight_t src_val, thrust::nullopt_t, weight_t w) const
  {
    auto push         = true;
    auto new_distance = src_val + w;
    auto threshold    = cutoff;
    if (vertex_partition.in_local_vertex_partition_range_nocheck(dst)) {
      auto local_vertex_offset =
        vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(dst);
      auto old_distance = *(distances + local_vertex_offset);
      threshold         = old_distance < threshold ? old_distance : threshold;
    }
    if (new_distance >= threshold) { push = false; }
    return push ? thrust::optional<thrust::tuple<weight_t, vertex_t>>{thrust::make_tuple(
                    new_distance, src)}
                : thrust::nullopt;
  }
};

}  // namespace

namespace detail {

template <typename GraphViewType, typename weight_t, typename PredecessorIterator>
void sssp(raft::handle_t const& handle,
          GraphViewType const& push_graph_view,
          edge_property_view_t<typename GraphViewType::edge_type, weight_t const*> edge_weight_view,
          weight_t* distances,
          PredecessorIterator predecessor_first,
          typename GraphViewType::vertex_type source_vertex,
          weight_t cutoff,
          bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = push_graph_view.number_of_vertices();
  auto const num_edges    = push_graph_view.number_of_edges();
  if (num_vertices == 0) { return; }

  // implements the Near-Far Pile method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.

  // 1. check input arguments

  CUGRAPH_EXPECTS(push_graph_view.is_valid_vertex(source_vertex),
                  "Invalid input argument: source vertex out-of-range.");

  if (do_expensive_check) {
    auto num_negative_edge_weights =
      count_if_e(handle,
                 push_graph_view,
                 edge_src_dummy_property_t{}.view(),
                 edge_dst_dummy_property_t{}.view(),
                 edge_weight_view,
                 [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w < 0.0; });
    CUGRAPH_EXPECTS(num_negative_edge_weights == 0,
                    "Invalid input argument: input edge weights should have non-negative values.");
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<weight_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  auto val_first = thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first));
  thrust::transform(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(push_graph_view.local_vertex_partition_range_first()),
    thrust::make_counting_iterator(push_graph_view.local_vertex_partition_range_last()),
    val_first,
    [source_vertex] __device__(auto val) {
      auto distance = invalid_distance;
      if (val == source_vertex) { distance = weight_t{0.0}; }
      return thrust::make_tuple(distance, invalid_vertex);
    });

  if (num_edges == 0) { return; }

  // 3. update delta

  weight_t average_vertex_degree{0.0};
  weight_t average_edge_weight{0.0};
  thrust::tie(average_vertex_degree, average_edge_weight) = transform_reduce_e(
    handle,
    push_graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_weight_view,
    [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) {
      return thrust::make_tuple(weight_t{1.0}, w);
    },
    thrust::make_tuple(weight_t{0.0}, weight_t{0.0}));
  average_vertex_degree /= static_cast<weight_t>(num_vertices);
  average_edge_weight /= static_cast<weight_t>(num_edges);
  auto delta =
    (static_cast<weight_t>(raft::warp_size()) * average_edge_weight) / average_vertex_degree;

  // 4. initialize SSSP frontier

  constexpr size_t bucket_idx_cur_near  = 0;
  constexpr size_t bucket_idx_next_near = 1;
  constexpr size_t bucket_idx_far       = 2;
  constexpr size_t num_buckets          = 3;

  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                       num_buckets);

  // 5. SSSP iteration

  auto edge_src_distances =
    GraphViewType::is_multi_gpu
      ? edge_src_property_t<GraphViewType, weight_t>(handle, push_graph_view)
      : edge_src_property_t<GraphViewType, weight_t>(handle);
  if (GraphViewType::is_multi_gpu) {
    fill_edge_src_property(
      handle, push_graph_view, std::numeric_limits<weight_t>::max(), edge_src_distances);
  }

  if (push_graph_view.in_local_vertex_partition_range_nocheck(source_vertex)) {
    vertex_frontier.bucket(bucket_idx_cur_near).insert(source_vertex);
  }

  auto near_far_threshold = delta;
  while (true) {
    if (GraphViewType::is_multi_gpu) {
      update_edge_src_property(handle,
                               push_graph_view,
                               vertex_frontier.bucket(bucket_idx_cur_near).begin(),
                               vertex_frontier.bucket(bucket_idx_cur_near).end(),
                               distances,
                               edge_src_distances);
    }

    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      push_graph_view.local_vertex_partition_view());

    auto [new_frontier_vertex_buffer, distance_predecessor_buffer] =
      transform_reduce_v_frontier_outgoing_e_by_dst(
        handle,
        push_graph_view,
        vertex_frontier.bucket(bucket_idx_cur_near),
        GraphViewType::is_multi_gpu
          ? edge_src_distances.view()
          : detail::edge_major_property_view_t<vertex_t, weight_t const*>(distances),
        edge_dst_dummy_property_t{}.view(),
        edge_weight_view,
        e_op_t<vertex_t, weight_t, GraphViewType::is_multi_gpu>{
          vertex_partition, distances, cutoff},
        reduce_op::minimum<thrust::tuple<weight_t, vertex_t>>());

    update_v_frontier(
      handle,
      push_graph_view,
      std::move(new_frontier_vertex_buffer),
      std::move(distance_predecessor_buffer),
      vertex_frontier,
      std::vector<size_t>{bucket_idx_next_near, bucket_idx_far},
      distances,
      thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
      [near_far_threshold] __device__(auto v, auto v_val, auto pushed_val) {
        auto new_dist = thrust::get<0>(pushed_val);
        auto update   = (new_dist < v_val);
        return thrust::make_tuple(
          update ? thrust::optional<size_t>{new_dist < near_far_threshold ? bucket_idx_next_near
                                                                          : bucket_idx_far}
                 : thrust::nullopt,
          update ? thrust::optional<thrust::tuple<weight_t, vertex_t>>{pushed_val}
                 : thrust::nullopt);
      });

    vertex_frontier.bucket(bucket_idx_cur_near).clear();
    vertex_frontier.bucket(bucket_idx_cur_near).shrink_to_fit();
    if (vertex_frontier.bucket(bucket_idx_next_near).aggregate_size() > 0) {
      vertex_frontier.swap_buckets(bucket_idx_cur_near, bucket_idx_next_near);
    } else if (vertex_frontier.bucket(bucket_idx_far).aggregate_size() >
               0) {  // near queue is empty, split the far queue
      auto old_near_far_threshold = near_far_threshold;
      near_far_threshold += delta;

      size_t near_size{0};
      size_t far_size{0};
      while (true) {
        vertex_frontier.split_bucket(
          bucket_idx_far,
          std::vector<size_t>{bucket_idx_cur_near},
          [vertex_partition, distances, old_near_far_threshold, near_far_threshold] __device__(
            auto v) {
            auto dist =
              *(distances + vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v));
            return dist >= old_near_far_threshold
                     ? thrust::optional<size_t>{dist < near_far_threshold ? bucket_idx_cur_near
                                                                          : bucket_idx_far}
                     : thrust::nullopt;
          });
        near_size = vertex_frontier.bucket(bucket_idx_cur_near).aggregate_size();
        far_size  = vertex_frontier.bucket(bucket_idx_far).aggregate_size();
        if ((near_size > 0) || (far_size == 0)) {
          break;
        } else {
          near_far_threshold += delta;
        }
      }
      if ((near_size == 0) && (far_size == 0)) { break; }
    } else {
      break;
    }
  }
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void sssp(raft::handle_t const& handle,
          graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
          edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
          weight_t* distances,
          vertex_t* predecessors,
          vertex_t source_vertex,
          weight_t cutoff,
          bool do_expensive_check)
{
  if (predecessors != nullptr) {
    detail::sssp(handle,
                 graph_view,
                 edge_weight_view,
                 distances,
                 predecessors,
                 source_vertex,
                 cutoff,
                 do_expensive_check);
  } else {
    detail::sssp(handle,
                 graph_view,
                 edge_weight_view,
                 distances,
                 thrust::make_discard_iterator(),
                 source_vertex,
                 cutoff,
                 do_expensive_check);
  }
}

}  // namespace cugraph

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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/count_if_e.cuh>
#include <cugraph/prims/edge_partition_src_dst_property.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>
#include <cugraph/prims/update_frontier_v_push_if_out_nbr.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/cudart_utils.h>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

namespace cugraph {
namespace detail {

template <typename GraphViewType, typename PredecessorIterator>
void sssp(raft::handle_t const& handle,
          GraphViewType const& push_graph_view,
          typename GraphViewType::weight_type* distances,
          PredecessorIterator predecessor_first,
          typename GraphViewType::vertex_type source_vertex,
          typename GraphViewType::weight_type cutoff,
          bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

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
  CUGRAPH_EXPECTS(push_graph_view.is_weighted(),
                  "Invalid input argument: an unweighted graph is passed to SSSP, BFS is more "
                  "efficient for unweighted graphs.");

  if (do_expensive_check) {
    auto num_negative_edge_weights =
      count_if_e(handle,
                 push_graph_view,
                 dummy_property_t<vertex_t>{}.device_view(),
                 dummy_property_t<vertex_t>{}.device_view(),
                 [] __device__(vertex_t, vertex_t, weight_t w, auto, auto) { return w < 0.0; });
    CUGRAPH_EXPECTS(num_negative_edge_weights == 0,
                    "Invalid input argument: input graph should have non-negative edge weights.");
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
    dummy_property_t<vertex_t>{}.device_view(),
    dummy_property_t<vertex_t>{}.device_view(),
    [] __device__(vertex_t, vertex_t, weight_t w, auto, auto) {
      return thrust::make_tuple(weight_t{1.0}, w);
    },
    thrust::make_tuple(weight_t{0.0}, weight_t{0.0}));
  average_vertex_degree /= static_cast<weight_t>(num_vertices);
  average_edge_weight /= static_cast<weight_t>(num_edges);
  auto delta =
    (static_cast<weight_t>(raft::warp_size()) * average_edge_weight) / average_vertex_degree;

  // 4. initialize SSSP frontier

  enum class Bucket { cur_near, next_near, far, num_buckets };
  VertexFrontier<vertex_t,
                 void,
                 GraphViewType::is_multi_gpu,
                 static_cast<size_t>(Bucket::num_buckets)>
    vertex_frontier(handle);

  // 5. SSSP iteration

  auto edge_partition_src_distances =
    GraphViewType::is_multi_gpu
      ? edge_partition_src_property_t<GraphViewType, weight_t>(handle, push_graph_view)
      : edge_partition_src_property_t<GraphViewType, weight_t>(handle);
  if (GraphViewType::is_multi_gpu) {
    edge_partition_src_distances.fill(std::numeric_limits<weight_t>::max(), handle.get_stream());
  }

  if (push_graph_view.in_local_vertex_partition_range_nocheck(source_vertex)) {
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).insert(source_vertex);
  }

  auto near_far_threshold = delta;
  while (true) {
    if (GraphViewType::is_multi_gpu) {
      update_edge_partition_src_property(
        handle,
        push_graph_view,
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).begin(),
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).end(),
        distances,
        edge_partition_src_distances);
    }

    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      push_graph_view.local_vertex_partition_view());

    update_frontier_v_push_if_out_nbr(
      handle,
      push_graph_view,
      vertex_frontier,
      static_cast<size_t>(Bucket::cur_near),
      std::vector<size_t>{static_cast<size_t>(Bucket::next_near), static_cast<size_t>(Bucket::far)},
      GraphViewType::is_multi_gpu
        ? edge_partition_src_distances.device_view()
        : detail::edge_partition_major_property_device_view_t<vertex_t, weight_t const*>(distances),
      dummy_property_t<vertex_t>{}.device_view(),
      [vertex_partition, distances, cutoff] __device__(
        vertex_t src, vertex_t dst, weight_t w, auto src_val, auto) {
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
      },
      reduce_op::min<thrust::tuple<weight_t, vertex_t>>(),
      distances,
      thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
      [near_far_threshold] __device__(auto v, auto v_val, auto pushed_val) {
        auto new_dist = thrust::get<0>(pushed_val);
        auto idx      = new_dist < v_val
                          ? (new_dist < near_far_threshold ? static_cast<size_t>(Bucket::next_near)
                                                           : static_cast<size_t>(Bucket::far))
                          : VertexFrontier<vertex_t>::kInvalidBucketIdx;
        return new_dist < v_val
                 ? thrust::optional<thrust::tuple<size_t, decltype(pushed_val)>>{thrust::make_tuple(
                     static_cast<size_t>(new_dist < near_far_threshold ? Bucket::next_near
                                                                       : Bucket::far),
                     pushed_val)}
                 : thrust::nullopt;
      });

    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).clear();
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).shrink_to_fit();
    if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next_near)).aggregate_size() > 0) {
      vertex_frontier.swap_buckets(static_cast<size_t>(Bucket::cur_near),
                                   static_cast<size_t>(Bucket::next_near));
    } else if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::far)).aggregate_size() >
               0) {  // near queue is empty, split the far queue
      auto old_near_far_threshold = near_far_threshold;
      near_far_threshold += delta;

      size_t near_size{0};
      size_t far_size{0};
      while (true) {
        vertex_frontier.split_bucket(
          static_cast<size_t>(Bucket::far),
          std::vector<size_t>{static_cast<size_t>(Bucket::cur_near)},
          [vertex_partition, distances, old_near_far_threshold, near_far_threshold] __device__(
            auto v) {
            auto dist =
              *(distances + vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v));
            return dist >= old_near_far_threshold
                     ? thrust::optional<size_t>{static_cast<size_t>(
                         dist < near_far_threshold ? Bucket::cur_near : Bucket::far)}
                     : thrust::nullopt;
          });
        near_size =
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).aggregate_size();
        far_size = vertex_frontier.get_bucket(static_cast<size_t>(Bucket::far)).aggregate_size();
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

  RAFT_CUDA_TRY(cudaStreamSynchronize(
    handle.get_stream()));  // this is as necessary vertex_frontier will become out-of-scope once
                            // this function returns (FIXME: should I stream sync in VertexFrontier
                            // destructor?)
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void sssp(raft::handle_t const& handle,
          graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
          weight_t* distances,
          vertex_t* predecessors,
          vertex_t source_vertex,
          weight_t cutoff,
          bool do_expensive_check)
{
  if (predecessors != nullptr) {
    detail::sssp(
      handle, graph_view, distances, predecessors, source_vertex, cutoff, do_expensive_check);
  } else {
    detail::sssp(handle,
                 graph_view,
                 distances,
                 thrust::make_discard_iterator(),
                 source_vertex,
                 cutoff,
                 do_expensive_check);
  }
}

}  // namespace cugraph

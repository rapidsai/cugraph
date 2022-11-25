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

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <prims/update_v_frontier.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace {

template <typename vertex_t, bool multi_gpu>
struct e_op_t {
  std::conditional_t<multi_gpu,
                     detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint8_t*>,
                     uint32_t*>
    visited_flags{};
  uint32_t const* prev_visited_flags{
    nullptr};  // relevant only if multi_gpu is false (this affects only local-computing with 0
               // impact in communication volume, so this may improve performance in small-scale but
               // will eat-up more memory with no benefit in performance in large-scale).
  vertex_t dst_first{};  // relevant only if multi_gpu is true

  __device__ thrust::optional<vertex_t> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    bool push{};
    if constexpr (multi_gpu) {
      auto dst_offset = dst - dst_first;
      auto old        = atomicOr(visited_flags.get_iter(dst_offset), uint8_t{1});
      push            = (old == uint8_t{0});
    } else {
      auto mask = uint32_t{1} << (dst % (sizeof(uint32_t) * 8));
      if (*(prev_visited_flags + (dst / (sizeof(uint32_t) * 8))) &
          mask) {  // check if unvisited in previous iterations
        push = false;
      } else {  // check if unvisited in this iteration as well
        auto old = atomicOr(visited_flags + (dst / (sizeof(uint32_t) * 8)), mask);
        push     = ((old & mask) == 0);
      }
    }
    return push ? thrust::optional<vertex_t>{src} : thrust::nullopt;
  }
};

}  // namespace

namespace detail {

template <typename GraphViewType, typename PredecessorIterator>
void bfs(raft::handle_t const& handle,
         GraphViewType const& push_graph_view,
         typename GraphViewType::vertex_type* distances,
         PredecessorIterator predecessor_first,
         typename GraphViewType::vertex_type const* sources,
         size_t n_sources,
         bool direction_optimizing,
         typename GraphViewType::vertex_type depth_limit,
         bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = push_graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS((n_sources == 0) || (sources != nullptr),
                  "Invalid input argument: sources cannot be null");

  auto aggregate_n_sources =
    GraphViewType::is_multi_gpu
      ? host_scalar_allreduce(
          handle.get_comms(), n_sources, raft::comms::op_t::SUM, handle.get_stream())
      : n_sources;
  CUGRAPH_EXPECTS(aggregate_n_sources > 0,
                  "Invalid input argument: input should have at least one source");

  CUGRAPH_EXPECTS(
    push_graph_view.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");

  if (do_expensive_check) {
    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      push_graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       sources,
                       sources + n_sources,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<vertex_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  thrust::fill(rmm::exec_policy(handle.get_thrust_policy()),
               distances,
               distances + push_graph_view.local_vertex_partition_range_size(),
               invalid_distance);
  thrust::fill(rmm::exec_policy(handle.get_thrust_policy()),
               predecessor_first,
               predecessor_first + push_graph_view.local_vertex_partition_range_size(),
               invalid_vertex);
  auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
    push_graph_view.local_vertex_partition_view());
  if (n_sources) {
    thrust::for_each(
      rmm::exec_policy(handle.get_thrust_policy()),
      sources,
      sources + n_sources,
      [vertex_partition, distances, predecessor_first] __device__(auto v) {
        *(distances + vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)) =
          vertex_t{0};
      });
  }

  // 3. initialize BFS frontier

  constexpr size_t bucket_idx_cur  = 0;
  constexpr size_t bucket_idx_next = 1;
  constexpr size_t num_buckets     = 2;

  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                       num_buckets);

  vertex_frontier.bucket(bucket_idx_cur).insert(sources, sources + n_sources);
  rmm::device_uvector<uint32_t> visited_flags(
    (push_graph_view.local_vertex_partition_range_size() + (sizeof(uint32_t) * 8 - 1)) /
      (sizeof(uint32_t) * 8),
    handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), visited_flags.begin(), visited_flags.end(), uint32_t{0});
  rmm::device_uvector<uint32_t> prev_visited_flags(
    GraphViewType::is_multi_gpu ? size_t{0} : visited_flags.size(),
    handle.get_stream());  // relevant only if GraphViewType::is_multi_gpu is false
  auto dst_visited_flags = GraphViewType::is_multi_gpu
                             ? edge_dst_property_t<GraphViewType, uint8_t>(handle, push_graph_view)
                             : edge_dst_property_t<GraphViewType,
                                                   uint8_t>(
                                 handle);  // relevant only if GraphViewType::is_multi_gpu is true
  if constexpr (GraphViewType::is_multi_gpu) {
    fill_edge_dst_property(handle, push_graph_view, uint8_t{0}, dst_visited_flags);
  }

  // 4. BFS iteration
  vertex_t depth{0};
  while (true) {
    if (direction_optimizing) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      if (GraphViewType::is_multi_gpu) {
        update_edge_dst_property(handle,
                                 push_graph_view,
                                 vertex_frontier.bucket(bucket_idx_cur).begin(),
                                 vertex_frontier.bucket(bucket_idx_cur).end(),
                                 thrust::make_constant_iterator(uint8_t{1}),
                                 dst_visited_flags);
      } else {
        thrust::copy(handle.get_thrust_policy(),
                     visited_flags.begin(),
                     visited_flags.end(),
                     prev_visited_flags.begin());
      }

      e_op_t<vertex_t, GraphViewType::is_multi_gpu> e_op{};
      if constexpr (GraphViewType::is_multi_gpu) {
        e_op.visited_flags =
          detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint8_t*>(
            dst_visited_flags.mutable_view());
        e_op.dst_first = push_graph_view.local_edge_partition_dst_range_first();
      } else {
        e_op.visited_flags      = visited_flags.data();
        e_op.prev_visited_flags = prev_visited_flags.data();
      }

      auto [new_frontier_vertex_buffer, predecessor_buffer] =
        transform_reduce_v_frontier_outgoing_e_by_dst(handle,
                                                      push_graph_view,
                                                      vertex_frontier.bucket(bucket_idx_cur),
                                                      edge_src_dummy_property_t{}.view(),
                                                      edge_dst_dummy_property_t{}.view(),
                                                      edge_dummy_property_t{}.view(),
#if 1
                                                      e_op,
#else
          // FIXME: need to test more about the performance trade-offs between additional
          // communication in updating dst_visited_flags (+ using atomics) vs reduced number of
          // pushes (leading to both less computation & communication in reduction)
          [vertex_partition, distances] __device__(
            vertex_t src, vertex_t dst, auto, auto, auto) {
            auto push = true;
            if (vertex_partition.in_local_vertex_partition_range_nocheck(dst)) {
              auto distance =
                *(distances +
                  vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(dst));
              if (distance != invalid_distance) { push = false; }
            }
            return thrust::make_tuple(push, src);
          },
#endif
                                                      reduce_op::any<vertex_t>());

      update_v_frontier(
        handle,
        push_graph_view,
        std::move(new_frontier_vertex_buffer),
        std::move(predecessor_buffer),
        vertex_frontier,
        std::vector<size_t>{bucket_idx_next},
        distances,
        thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
        [depth] __device__(auto v, auto v_val, auto pushed_val) {
          auto update = (v_val == invalid_distance);
          return thrust::make_tuple(
            update ? thrust::optional<size_t>{bucket_idx_next} : thrust::nullopt,
            update ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(
                       depth + 1, pushed_val)}
                   : thrust::nullopt);
        });

      vertex_frontier.bucket(bucket_idx_cur).clear();
      vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
      vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
      if (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() == 0) { break; }
    }

    depth++;
    if (depth >= depth_limit) { break; }
  }
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
void bfs(raft::handle_t const& handle,
         graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
         vertex_t* distances,
         vertex_t* predecessors,
         vertex_t const* sources,
         size_t n_sources,
         bool direction_optimizing,
         vertex_t depth_limit,
         bool do_expensive_check)
{
  if (predecessors != nullptr) {
    detail::bfs(handle,
                graph_view,
                distances,
                predecessors,
                sources,
                n_sources,
                direction_optimizing,
                depth_limit,
                do_expensive_check);
  } else {
    detail::bfs(handle,
                graph_view,
                distances,
                thrust::make_discard_iterator(),
                sources,
                n_sources,
                direction_optimizing,
                depth_limit,
                do_expensive_check);
  }
}

}  // namespace cugraph

/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_v_frontier_outgoing_e_by_src_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

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
struct topdown_e_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool> visited_flags{};
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
      auto old        = visited_flags.atomic_or(dst_offset, true);
      push            = !old;
    } else {
      if (*(prev_visited_flags + packed_bool_offset(dst)) &
          packed_bool_mask(dst)) {  // check if unvisited in previous iterations
        push = false;
      } else {  // check if unvisited in this iteration as well
        auto old = visited_flags.atomic_or(dst, true);
        push     = !old;
      }
    }
    return push ? thrust::optional<vertex_t>{src} : thrust::nullopt;
  }
};

template <typename vertex_t, bool multi_gpu>
struct bottomup_e_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool> visited_flags{};
  vertex_t dst_first{};  // relevant only if multi_gpu is true

  __device__ thrust::optional<vertex_t> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    vertex_t dst_offset;
    if constexpr (multi_gpu) {
      dst_offset = dst - dst_first;
    } else {
      dst_offset = dst;
    }
    auto push = visited_flags.get(dst_offset);
    return push ? thrust::optional<vertex_t>{dst} : thrust::nullopt;
  }
};

}  // namespace

namespace detail {

template <typename GraphViewType, typename PredecessorIterator>
void bfs(raft::handle_t const& handle,
         GraphViewType const& graph_view,
         typename GraphViewType::vertex_type* distances,
         PredecessorIterator predecessor_first,
         typename GraphViewType::vertex_type const* sources,
         size_t n_sources,
         bool direction_optimizing,
         typename GraphViewType::vertex_type depth_limit,
         bool do_expensive_check)
{
  direction_optimizing = false;  // FIXME
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  // direction optimizing BFS implementation is based on "S. Beamer, K. Asanovic, D. Patterson,
  // Direction-Optimizing Breadth-First Search, 2012"

  auto const num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS((n_sources == 0) || (sources != nullptr),
                  "Invalid input argument: sources cannot be null if n_sources > 0.");

  auto aggregate_n_sources = GraphViewType::is_multi_gpu
                               ? host_scalar_allreduce(handle.get_comms(),
                                                       static_cast<vertex_t>(n_sources),
                                                       raft::comms::op_t::SUM,
                                                       handle.get_stream())
                               : static_cast<vertex_t>(n_sources);
  CUGRAPH_EXPECTS(aggregate_n_sources > 0,
                  "Invalid input argument: input should have at least one source");

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");

  auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
    graph_view.local_vertex_partition_view());

  if (do_expensive_check) {
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

  thrust::fill(handle.get_thrust_policy(),
               distances,
               distances + graph_view.local_vertex_partition_range_size(),
               invalid_distance);
  thrust::fill(handle.get_thrust_policy(),
               predecessor_first,
               predecessor_first + graph_view.local_vertex_partition_range_size(),
               invalid_vertex);
  if (n_sources > 0) {
    thrust::for_each(
      handle.get_thrust_policy(),
      sources,
      sources + n_sources,
      [vertex_partition, distances, predecessor_first] __device__(auto v) {
        *(distances + vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)) =
          vertex_t{0};
      });
  }

  // 3. update meta data for direction optimizing BFS

  constexpr edge_t direction_optimizing_alpha  = 14;
  constexpr vertex_t direction_optimizing_beta = 24;

  std::optional<rmm::device_uvector<edge_t>> out_degrees{std::nullopt};
  // FIXME: can we remove false positives if we exclude the vertices in the source list?
  std::optional<rmm::device_uvector<vertex_t>> nzd_possibly_unvisited_vertices{
    std::nullopt};  // this contains the entire set of (local) unvisited vertices + some (local) visited vertices
  if (direction_optimizing) {
    out_degrees = graph_view.compute_out_degrees(handle);
    // FIXME: actually, we know the vertex range for non-zero-degree vertices in advance (due to our
    // renumbering scheme). Check the performance difference to see whether it is worthwhile to
    // exploit this implementation details.
    // FIXME: we also need to double check compute_out_degrees() is exploiting this information.
    nzd_possibly_unvisited_vertices = rmm::device_uvector<vertex_t>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());
    (*nzd_possibly_unvisited_vertices)
      .resize(thrust::distance(
                (*nzd_possibly_unvisited_vertices).begin(),
                thrust::copy_if(
                  handle.get_thrust_policy(),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
                  (*out_degrees).begin(),
                  (*nzd_possibly_unvisited_vertices).begin(),
                  [] __device__(edge_t out_degree) { return (out_degree > edge_t{0}); })),
              handle.get_stream());
    (*nzd_possibly_unvisited_vertices).shrink_to_fit(handle.get_stream());
  }

  // 4. initialize BFS frontier

  constexpr size_t bucket_idx_cur  = 0;
  constexpr size_t bucket_idx_next = 1;
  constexpr size_t num_buckets     = 2;

  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                       num_buckets);

  vertex_frontier.bucket(bucket_idx_cur).insert(sources, sources + n_sources);
  rmm::device_uvector<uint32_t> visited_flags(
    packed_bool_size(graph_view.local_vertex_partition_range_size()),
    handle.get_stream());  // FIXME: relevant only if GraphViewType::is_multi_gpu is false??? and
                           // should I better rename this to surely_visited_flags? (not all visited
                           // vertices are marked to true) or better update this properly... i.e. set true for sources
  thrust::fill(handle.get_thrust_policy(),
               visited_flags.begin(),
               visited_flags.end(),
               packed_bool_empty_mask());
  rmm::device_uvector<uint32_t> prev_visited_flags(
    GraphViewType::is_multi_gpu ? size_t{0} : visited_flags.size(),
    handle.get_stream());  // relevant only if GraphViewType::is_multi_gpu is false
  // FIXME: better be std::conditional?
  auto dst_visited_flags =
    GraphViewType::is_multi_gpu
      ? edge_dst_property_t<GraphViewType, bool>(handle, graph_view)
      : edge_dst_property_t<GraphViewType,
                            bool>(handle);  // relevant only if GraphViewType::is_multi_gpu is true
  if constexpr (GraphViewType::is_multi_gpu) {
    fill_edge_dst_property(handle, graph_view, false, dst_visited_flags);
  }

  if (GraphViewType::is_multi_gpu) {
    update_edge_dst_property(handle,
                             graph_view,
                             vertex_frontier.bucket(bucket_idx_cur).begin(),
                             vertex_frontier.bucket(bucket_idx_cur).end(),
                             thrust::make_constant_iterator(true),
                             dst_visited_flags);
  } else {
    // FIMXE: unnecessary if multi_gpu?
    thrust::for_each(handle.get_thrust_policy(),
                     sources,
                     sources + n_sources,
                     [vertex_partition,
                      visited_flags = raft::device_span<uint32_t>(
                        visited_flags.data(), visited_flags.size())] __device__(auto v) {
                       auto v_offset =
                         vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                       cuda::atomic_ref<uint32_t> mask(visited_flags[packed_bool_offset(v_offset)]);
                       mask.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
                     });
    thrust::copy(handle.get_thrust_policy(),
                 visited_flags.begin(),
                 visited_flags.end(),
                 prev_visited_flags.begin());
  }

  // 4. BFS iteration
  vertex_t depth{0};
  bool top_down                           = true;
  auto cur_aggregate_vertex_frontier_size = aggregate_n_sources;
  while (true) {
    vertex_t next_aggregate_vertex_frontier_size{};
    if (top_down) {
      topdown_e_op_t<vertex_t, GraphViewType::is_multi_gpu> e_op{};
      if constexpr (GraphViewType::is_multi_gpu) {
        e_op.visited_flags =
          detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
            dst_visited_flags.mutable_view());
        e_op.dst_first = graph_view.local_edge_partition_dst_range_first();
      } else {
        e_op.visited_flags =
          detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
            detail::edge_minor_property_view_t<vertex_t, uint32_t*, bool>(visited_flags.data(),
                                                                          vertex_t{0}));
        e_op.prev_visited_flags = prev_visited_flags.data();
      }

      auto [new_frontier_vertex_buffer, predecessor_buffer] =
        transform_reduce_v_frontier_outgoing_e_by_dst(handle,
                                                      graph_view,
                                                      vertex_frontier.bucket(bucket_idx_cur),
                                                      edge_src_dummy_property_t{}.view(),
                                                      edge_dst_dummy_property_t{}.view(),
                                                      edge_dummy_property_t{}.view(),
                                                      e_op,
                                                      reduce_op::any<vertex_t>());

      std::cout << "topdown frontier_size=" << vertex_frontier.bucket(bucket_idx_cur).size() << " new_frontier_size=" << new_frontier_vertex_buffer.size() << std::endl;
      update_v_frontier(
        handle,
        graph_view,
        std::move(new_frontier_vertex_buffer),
        std::move(predecessor_buffer),
        vertex_frontier,
        std::vector<size_t>{bucket_idx_next},
        distances,
        thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
        [depth] __device__(auto v, auto v_val, auto pushed_val) {
          // FIXME: should I check this?
          auto update = (v_val == invalid_distance);
          return thrust::make_tuple(
            update ? thrust::optional<size_t>{bucket_idx_next} : thrust::nullopt,
            update ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(
                       depth + 1, pushed_val)}
                   : thrust::nullopt);
        });

      next_aggregate_vertex_frontier_size =
        static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_next).aggregate_size());
      if (next_aggregate_vertex_frontier_size == 0) { break; }

      if (direction_optimizing) {
        auto m_f = thrust::transform_reduce(
          handle.get_thrust_policy(),
          vertex_frontier.bucket(bucket_idx_next).begin(),
          vertex_frontier.bucket(bucket_idx_next).end(),
          [vertex_partition,
           out_degrees = raft::device_span<edge_t const>(
             (*out_degrees).data(), (*out_degrees).size())] __device__(vertex_t v) {
            auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            return out_degrees[v_offset];
          },
          edge_t{0},
          thrust::plus<edge_t>{});
        (*nzd_possibly_unvisited_vertices)
          .resize(thrust::distance(
                    (*nzd_possibly_unvisited_vertices).begin(),
                    thrust::remove_if(
                      handle.get_thrust_policy(),
                      (*nzd_possibly_unvisited_vertices).begin(),
                      (*nzd_possibly_unvisited_vertices).end(),
                      // FIXME: checking visited_flags might be faster
                      [vertex_partition, distances] __device__(vertex_t v) {
                        auto v_offset =
                          vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                        return distances[v_offset] != invalid_distance;
                      })),
                  handle.get_stream());
        auto m_u = thrust::transform_reduce(
          handle.get_thrust_policy(),
          (*nzd_possibly_unvisited_vertices).begin(),
          (*nzd_possibly_unvisited_vertices).end(),
          [vertex_partition,
           out_degrees = raft::device_span<edge_t const>(
             (*out_degrees).data(), (*out_degrees).size())] __device__(vertex_t v) {
            auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            return out_degrees[v_offset];
          },
          edge_t{0},
          thrust::plus<edge_t>{});
        std::cout << "nzd_possibly_unvisited_vertices.size()=" << (*nzd_possibly_unvisited_vertices).size() << " m_f=" << m_f << " m_u=" << m_u << std::endl;
        if ((m_f > m_u * direction_optimizing_alpha) &&
            (next_aggregate_vertex_frontier_size >= cur_aggregate_vertex_frontier_size)) {
          top_down = false;
        }
      }

      if (top_down) {  // staying in top-down
        vertex_frontier.bucket(bucket_idx_cur).clear();
        vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
        vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
      } else {  // swithcing to bottom-up
        vertex_frontier.bucket(bucket_idx_cur).clear();
        vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
        vertex_frontier.bucket(bucket_idx_next).clear();
        vertex_frontier.bucket(bucket_idx_next).shrink_to_fit();
        // FIXME: this copy could be avoided... maybe bucket_view???
        vertex_frontier.bucket(bucket_idx_cur)
          .insert((*nzd_possibly_unvisited_vertices).begin(),
                  (*nzd_possibly_unvisited_vertices).end());
      }

      if (GraphViewType::is_multi_gpu) {
        update_edge_dst_property(handle,
                                 graph_view,
                                 vertex_frontier.bucket(bucket_idx_cur).begin(),
                                 vertex_frontier.bucket(bucket_idx_cur).end(),
                                 thrust::make_constant_iterator(true),
                                 dst_visited_flags);
      } else {
        thrust::copy(handle.get_thrust_policy(),
                     visited_flags.begin(),
                     visited_flags.end(),
                     prev_visited_flags.begin());
      }
    } else {  // bottom up
      bottomup_e_op_t<vertex_t, GraphViewType::is_multi_gpu> e_op{};
      if constexpr (GraphViewType::is_multi_gpu) {
        e_op.visited_flags =
          detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
            dst_visited_flags.mutable_view());
        e_op.dst_first = graph_view.local_edge_partition_dst_range_first();
      } else {
        e_op.visited_flags =
          detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
            detail::edge_minor_property_view_t<vertex_t, uint32_t*, bool>(visited_flags.data(),
                                                                          vertex_t{0}));
      }
      auto [new_frontier_vertex_buffer, predecessor_buffer] =
        transform_reduce_v_frontier_outgoing_e_by_src(
          handle,
          graph_view,
          vertex_frontier.bucket(bucket_idx_cur),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          e_op,
          reduce_op::any<vertex_t>());  // FIXME: if reduce_op is any, break on first finding?
      std::cout << "bottomup frontier_size=" << vertex_frontier.bucket(bucket_idx_cur).aggregate_size()  << " new_frontier(found int this iteration)_size=" << new_frontier_vertex_buffer.size() << std::endl;

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(new_frontier_vertex_buffer.begin(), predecessor_buffer.begin()),
        thrust::make_zip_iterator(new_frontier_vertex_buffer.end(), predecessor_buffer.end()),
        [distances, predecessor_first, depth, vertex_partition] __device__(auto tup) {
          auto v_offset =
            vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(thrust::get<0>(tup));
          distances[v_offset]             = depth + 1;
          *(predecessor_first + v_offset) = thrust::get<1>(tup);
        });

      assert(direction_optimizing);

      (*nzd_possibly_unvisited_vertices)
        .resize(thrust::distance(
                  (*nzd_possibly_unvisited_vertices).begin(),
                  thrust::remove_if(
                    handle.get_thrust_policy(),
                    (*nzd_possibly_unvisited_vertices).begin(),
                    (*nzd_possibly_unvisited_vertices).end(),
                    [vertex_partition, distances] __device__(vertex_t v) {
                      auto v_offset =
                        vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                      // FIXME: using visited can be faster???
                      return distances[v_offset] != invalid_distance;
                    })),
                handle.get_stream());

      next_aggregate_vertex_frontier_size =
        GraphViewType::is_multi_gpu
          ? host_scalar_allreduce(handle.get_comms(),
                                  static_cast<vertex_t>(new_frontier_vertex_buffer.size()),
                                  raft::comms::op_t::SUM,
                                  handle.get_stream())
          : static_cast<vertex_t>(new_frontier_vertex_buffer.size());
      if (next_aggregate_vertex_frontier_size == 0) { break; }

      auto aggregate_nzd_possibly_unvisted_vertices =
        GraphViewType::is_multi_gpu
          ? host_scalar_allreduce(handle.get_comms(),
                                  static_cast<vertex_t>((*nzd_possibly_unvisited_vertices).size()),
                                  raft::comms::op_t::SUM,
                                  handle.get_stream())
          : static_cast<vertex_t>((*nzd_possibly_unvisited_vertices).size());

      std::cout << "next_aggregate_vertex_frontier_size=" << next_aggregate_vertex_frontier_size <<
        " aggregate_nzd_possibly_unvisted_vertices=" << aggregate_nzd_possibly_unvisted_vertices << std::endl;
      if ((next_aggregate_vertex_frontier_size * direction_optimizing_beta < aggregate_nzd_possibly_unvisted_vertices) &&
          (next_aggregate_vertex_frontier_size < cur_aggregate_vertex_frontier_size)) {
        top_down = true;
      }

      vertex_frontier.bucket(bucket_idx_cur).clear();
      if (top_down) {  // swithcing to top-down
        // FIXME: std::move(new_frontier_vertex_buffer) will be faster
        vertex_frontier.bucket(bucket_idx_cur)
          .insert(new_frontier_vertex_buffer.begin(), new_frontier_vertex_buffer.end());
      } else {  // staying in bottom-up
        // FIXME: this copy can be avoided.
        vertex_frontier.bucket(bucket_idx_cur)
          .insert((*nzd_possibly_unvisited_vertices).begin(),
                  (*nzd_possibly_unvisited_vertices).end());
      }

      if (GraphViewType::is_multi_gpu) {
        update_edge_dst_property(handle,
                                 graph_view,
                                 new_frontier_vertex_buffer.begin(),
                                 new_frontier_vertex_buffer.end(),
                                 thrust::make_constant_iterator(true),
                                 dst_visited_flags);
      } else {
        // FIXME: is this necessary? Can we perform this inside the op?
        thrust::for_each(
          handle.get_thrust_policy(),
          new_frontier_vertex_buffer.begin(),
          new_frontier_vertex_buffer.end(),
          [vertex_partition,
           visited_flags = raft::device_span<uint32_t>(visited_flags.data(),
                                                       visited_flags.size())] __device__(auto v) {
            auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            cuda::atomic_ref<uint32_t> mask(visited_flags[packed_bool_offset(v_offset)]);
            mask.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
          });
        // FIXME: unused if bottom_up
        thrust::copy(handle.get_thrust_policy(),
                     visited_flags.begin(),
                     visited_flags.end(),
                     prev_visited_flags.begin());
      }
    }
    cur_aggregate_vertex_frontier_size = next_aggregate_vertex_frontier_size;

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

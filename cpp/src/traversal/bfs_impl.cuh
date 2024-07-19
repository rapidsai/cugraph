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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace {

template <typename vertex_t, bool multi_gpu>
struct topdown_e_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>
    prev_visited_flags{};  // visited in the previous iterations, to reduce the number of atomic
                           // operations
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool> visited_flags{};
  vertex_t dst_first{};

  __device__ thrust::optional<vertex_t> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    auto dst_offset = dst - dst_first;
    auto old        = prev_visited_flags.get(dst_offset);
    if (!old) { old = visited_flags.atomic_or(dst_offset, true); }
    return old ? thrust::nullopt : thrust::optional<vertex_t>{src};
  }
};

template <typename vertex_t, bool multi_gpu>
struct bottomup_e_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>
    prev_visited_flags{};  // visited in the previous iterations
  vertex_t dst_first{};

  __device__ thrust::optional<vertex_t> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    auto dst_offset = dst - dst_first;
    auto old        = prev_visited_flags.get(dst_offset);
    return old ? thrust::optional<vertex_t>{dst} : thrust::nullopt;
  }
};

}  // namespace

namespace detail {

#if 1  // FIXME: delete
#define BFS_PERFORMANCE_MEASUREMENT 1
#endif

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
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto prep0 = std::chrono::steady_clock::now();
#endif
  // direction optimizing BFS implementation is based on "S. Beamer, K. Asanovic, D. Patterson,
  // Direction-Optimizing Breadth-First Search, 2012"

  auto const num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS((n_sources == 0) || (sources != nullptr),
                  "Invalid input argument: sources cannot be null if n_sources > 0.");

  if (GraphViewType::is_multi_gpu) {
    if (do_expensive_check) {
      auto aggregate_n_sources = host_scalar_allreduce(handle.get_comms(),
                                                       static_cast<vertex_t>(n_sources),
                                                       raft::comms::op_t::SUM,
                                                       handle.get_stream());
      CUGRAPH_EXPECTS(aggregate_n_sources > 0,
                      "Invalid input argument: input should have at least one source");
    }
  } else {
    CUGRAPH_EXPECTS(n_sources > 0,
                    "Invalid input argument: input should have at least one source.");
  }

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");

  auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
    graph_view.local_vertex_partition_view());

  if (do_expensive_check) {
    bool is_sorted = thrust::is_sorted(handle.get_thrust_policy(), sources, sources + n_sources);
    if constexpr (GraphViewType::is_multi_gpu) {
      is_sorted = static_cast<bool>(host_scalar_allreduce(handle.get_comms(),
                                                          static_cast<int32_t>(is_sorted),
                                                          raft::comms::op_t::SUM,
                                                          handle.get_stream()));
    }

    CUGRAPH_EXPECTS(
      is_sorted,
      "Invalid input arguments: input sources should be sorted in the non-descending order.");

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
  auto output_first = thrust::make_permutation_iterator(
    distances,
    thrust::make_transform_iterator(
      sources, detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}));
  thrust::fill(handle.get_thrust_policy(), output_first, output_first + n_sources, vertex_t{0});

  // 3. update meta data for direction optimizing BFS

  constexpr edge_t direction_optimizing_alpha  = 14;
  constexpr vertex_t direction_optimizing_beta = 24;

  std::optional<rmm::device_uvector<edge_t>> out_degrees{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> nzd_unvisited_vertices{std::nullopt};
  if (direction_optimizing) {
    out_degrees            = graph_view.compute_out_degrees(handle);
    nzd_unvisited_vertices = rmm::device_uvector<vertex_t>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());
    (*nzd_unvisited_vertices)
      .resize(thrust::distance(
                (*nzd_unvisited_vertices).begin(),
                thrust::copy_if(
                  handle.get_thrust_policy(),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
                  (*nzd_unvisited_vertices).begin(),
                  [vertex_partition,
                   sources     = raft::device_span<vertex_t const>(sources, n_sources),
                   out_degrees = raft::device_span<edge_t const>(
                     (*out_degrees).data(), (*out_degrees).size())] __device__(vertex_t v) {
                    auto v_offset =
                      vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                    return (out_degrees[v_offset] > edge_t{0}) &&
                           !thrust::binary_search(thrust::seq, sources.begin(), sources.end(), v);
                  })),
              handle.get_stream());
    (*nzd_unvisited_vertices).shrink_to_fit(handle.get_stream());
  }

  // 4. initialize BFS frontier

  constexpr size_t bucket_idx_cur  = 0;
  constexpr size_t bucket_idx_next = 1;
  constexpr size_t num_buckets     = 2;

  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                       num_buckets);
  vertex_frontier.bucket(bucket_idx_cur).insert(sources, sources + n_sources);

  // 5. initialize BFS temporary state data

  auto prev_dst_visited_flags = edge_dst_property_t<GraphViewType, bool>(handle, graph_view);
  fill_edge_dst_property(handle, graph_view, prev_dst_visited_flags.mutable_view(), false);
  auto dst_visited_flags = edge_dst_property_t<GraphViewType, bool>(
    handle, graph_view);  // this may mark some vertices visited in previous iterations as unvisited
                          // (but this is OK as we check prev_dst_visited_flags first)
  fill_edge_dst_property(handle, graph_view, dst_visited_flags.mutable_view(), false);

  fill_edge_dst_property(handle,
                         graph_view,
                         vertex_frontier.bucket(bucket_idx_cur).begin(),
                         vertex_frontier.bucket(bucket_idx_cur).end(),
                         prev_dst_visited_flags.mutable_view(),
                         true);
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto prep1                        = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur = prep1 - prep0;
  std::cout << "prep took " << dur.count() << " s." << std::endl;
#endif

  // 4. BFS iteration
  vertex_t depth{0};
  bool top_down = true;
  auto cur_aggregate_vertex_frontier_size =
    static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_cur).aggregate_size());
  while (true) {
    vertex_t next_aggregate_vertex_frontier_size{};
    if (top_down) {
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown0 = std::chrono::steady_clock::now();
#endif
      topdown_e_op_t<vertex_t, GraphViewType::is_multi_gpu> e_op{};
      e_op.prev_visited_flags =
        detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
          prev_dst_visited_flags.mutable_view());
      e_op.visited_flags =
        detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
          dst_visited_flags.mutable_view());
      e_op.dst_first = graph_view.local_edge_partition_dst_range_first();

      auto [new_frontier_vertex_buffer, predecessor_buffer] =
        transform_reduce_v_frontier_outgoing_e_by_dst(handle,
                                                      graph_view,
                                                      vertex_frontier.bucket(bucket_idx_cur),
                                                      edge_src_dummy_property_t{}.view(),
                                                      edge_dst_dummy_property_t{}.view(),
                                                      edge_dummy_property_t{}.view(),
                                                      e_op,
                                                      reduce_op::any<vertex_t>());
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown1 = std::chrono::steady_clock::now();
#endif

      auto input_pair_first = thrust::make_zip_iterator(thrust::make_constant_iterator(depth + 1),
                                                        predecessor_buffer.begin());
      thrust::scatter(
        handle.get_thrust_policy(),
        input_pair_first,
        input_pair_first + new_frontier_vertex_buffer.size(),
        thrust::make_transform_iterator(
          new_frontier_vertex_buffer.begin(),
          detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        thrust::make_zip_iterator(distances, predecessor_first));
      vertex_frontier.bucket(bucket_idx_next) =
        key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
          handle, std::move(new_frontier_vertex_buffer));
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown2 = std::chrono::steady_clock::now();
#endif

      next_aggregate_vertex_frontier_size =
        static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_next).aggregate_size());
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown3 = std::chrono::steady_clock::now();
#endif
      if (next_aggregate_vertex_frontier_size == 0) {
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
        std::chrono::duration<double> dur0 = topdown1 - topdown0;
        std::chrono::duration<double> dur1 = topdown2 - topdown1;
        std::chrono::duration<double> dur2 = topdown3 - topdown2;
        std::chrono::duration<double> dur  = topdown3 - topdown0;
        std::cout << "topdown took " << dur.count() << " (" << dur0.count() << "," << dur1.count()
                  << "," << dur2.count() << ") s." << std::endl;
#endif
        break;
      }

      fill_edge_dst_property(handle,
                             graph_view,
                             vertex_frontier.bucket(bucket_idx_next).begin(),
                             vertex_frontier.bucket(bucket_idx_next).end(),
                             prev_dst_visited_flags.mutable_view(),
                             true);
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown4 = std::chrono::steady_clock::now();
#endif

      if (direction_optimizing) {
        auto m_f = thrust::transform_reduce(
          handle.get_thrust_policy(),
          vertex_frontier.bucket(bucket_idx_next).begin(),
          vertex_frontier.bucket(bucket_idx_next).end(),
          cuda::proclaim_return_type<edge_t>(
            [vertex_partition,
             out_degrees = raft::device_span<edge_t const>(
               (*out_degrees).data(), (*out_degrees).size())] __device__(vertex_t v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
              return out_degrees[v_offset];
            }),
          edge_t{0},
          thrust::plus<edge_t>{});

        {
          rmm::device_uvector<vertex_t> tmp_vertices((*nzd_unvisited_vertices).size(),
                                                     handle.get_stream());
          tmp_vertices.resize(
            thrust::distance(tmp_vertices.begin(),
                             thrust::set_difference(handle.get_thrust_policy(),
                                                    (*nzd_unvisited_vertices).begin(),
                                                    (*nzd_unvisited_vertices).end(),
                                                    vertex_frontier.bucket(bucket_idx_next).begin(),
                                                    vertex_frontier.bucket(bucket_idx_next).end(),
                                                    tmp_vertices.begin())),
            handle.get_stream());
          nzd_unvisited_vertices = std::move(tmp_vertices);
        }

        auto m_u = thrust::transform_reduce(
          handle.get_thrust_policy(),
          (*nzd_unvisited_vertices).begin(),
          (*nzd_unvisited_vertices).end(),
          cuda::proclaim_return_type<edge_t>(
            [vertex_partition,
             out_degrees = raft::device_span<edge_t const>(
               (*out_degrees).data(), (*out_degrees).size())] __device__(vertex_t v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
              return out_degrees[v_offset];
            }),
          edge_t{0},
          thrust::plus<edge_t>{});
        auto aggregate_m_f =
          GraphViewType::is_multi_gpu
            ? host_scalar_allreduce(
                handle.get_comms(), m_f, raft::comms::op_t::SUM, handle.get_stream())
            : m_f;
        auto aggregate_m_u =
          GraphViewType::is_multi_gpu
            ? host_scalar_allreduce(
                handle.get_comms(), m_u, raft::comms::op_t::SUM, handle.get_stream())
            : m_u;
        if ((aggregate_m_f * direction_optimizing_alpha > aggregate_m_u) &&
            (next_aggregate_vertex_frontier_size >= cur_aggregate_vertex_frontier_size)) {
          top_down = false;
        }
      }
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown5 = std::chrono::steady_clock::now();
#endif

      if (top_down) {  // staying in top-down
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(handle);
        vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
      } else {  // swithcing to bottom-up
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
            handle,
            raft::device_span<vertex_t const>((*nzd_unvisited_vertices).data(),
                                              (*nzd_unvisited_vertices).size()));
        vertex_frontier.bucket(bucket_idx_next) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(handle);
      }
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto topdown6                      = std::chrono::steady_clock::now();
      std::chrono::duration<double> dur0 = topdown1 - topdown0;
      std::chrono::duration<double> dur1 = topdown2 - topdown1;
      std::chrono::duration<double> dur2 = topdown3 - topdown2;
      std::chrono::duration<double> dur3 = topdown4 - topdown3;
      std::chrono::duration<double> dur4 = topdown5 - topdown4;
      std::chrono::duration<double> dur5 = topdown6 - topdown5;
      std::chrono::duration<double> dur  = topdown6 - topdown0;
      std::cout << "topdown took " << dur.count() << " (" << dur0.count() << "," << dur1.count()
                << "," << dur2.count() << "," << dur3.count() << "," << dur4.count() << ","
                << dur5.count() << ") s." << std::endl;
#endif
    } else {                 // bottom up
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup0 = std::chrono::steady_clock::now();
#endif
      bottomup_e_op_t<vertex_t, GraphViewType::is_multi_gpu> e_op{};
      e_op.prev_visited_flags =
        detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
          prev_dst_visited_flags.mutable_view());
      e_op.dst_first = graph_view.local_edge_partition_dst_range_first();
      auto [new_frontier_vertex_buffer, predecessor_buffer] =
        transform_reduce_v_frontier_outgoing_e_by_src(handle,
                                                      graph_view,
                                                      vertex_frontier.bucket(bucket_idx_cur),
                                                      edge_src_dummy_property_t{}.view(),
                                                      edge_dst_dummy_property_t{}.view(),
                                                      edge_dummy_property_t{}.view(),
                                                      e_op,
                                                      reduce_op::any<vertex_t>());
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup1 = std::chrono::steady_clock::now();
#endif

      auto input_pair_first = thrust::make_zip_iterator(thrust::make_constant_iterator(depth + 1),
                                                        predecessor_buffer.begin());
      thrust::scatter(
        handle.get_thrust_policy(),
        input_pair_first,
        input_pair_first + new_frontier_vertex_buffer.size(),
        thrust::make_transform_iterator(
          new_frontier_vertex_buffer.begin(),
          detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        thrust::make_zip_iterator(distances, predecessor_first));

      assert(direction_optimizing);
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup2 = std::chrono::steady_clock::now();
#endif

      {
        rmm::device_uvector<vertex_t> tmp_vertices((*nzd_unvisited_vertices).size(),
                                                   handle.get_stream());
        tmp_vertices.resize(
          thrust::distance(tmp_vertices.begin(),
                           thrust::set_difference(handle.get_thrust_policy(),
                                                  (*nzd_unvisited_vertices).begin(),
                                                  (*nzd_unvisited_vertices).end(),
                                                  new_frontier_vertex_buffer.begin(),
                                                  new_frontier_vertex_buffer.end(),
                                                  tmp_vertices.begin())),
          handle.get_stream());
        nzd_unvisited_vertices = std::move(tmp_vertices);
      }
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup3 = std::chrono::steady_clock::now();
#endif

      next_aggregate_vertex_frontier_size =
        GraphViewType::is_multi_gpu
          ? host_scalar_allreduce(handle.get_comms(),
                                  static_cast<vertex_t>(new_frontier_vertex_buffer.size()),
                                  raft::comms::op_t::SUM,
                                  handle.get_stream())
          : static_cast<vertex_t>(new_frontier_vertex_buffer.size());
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup4 = std::chrono::steady_clock::now();
#endif
      if (next_aggregate_vertex_frontier_size == 0) {
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
        std::chrono::duration<double> dur0 = bottomup1 - bottomup0;
        std::chrono::duration<double> dur1 = bottomup2 - bottomup1;
        std::chrono::duration<double> dur2 = bottomup3 - bottomup2;
        std::chrono::duration<double> dur3 = bottomup4 - bottomup3;
        std::chrono::duration<double> dur  = bottomup4 - bottomup0;
        std::cout << "bottomup took " << dur.count() << " (" << dur0.count() << "," << dur1.count()
                  << "," << dur2.count() << "," << dur3.count() << " s." << std::endl;
#endif
        break;
      }

      fill_edge_dst_property(handle,
                             graph_view,
                             new_frontier_vertex_buffer.begin(),
                             new_frontier_vertex_buffer.end(),
                             prev_dst_visited_flags.mutable_view(),
                             true);
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup5 = std::chrono::steady_clock::now();
#endif

      auto aggregate_nzd_unvisted_vertices =
        GraphViewType::is_multi_gpu
          ? host_scalar_allreduce(handle.get_comms(),
                                  static_cast<vertex_t>((*nzd_unvisited_vertices).size()),
                                  raft::comms::op_t::SUM,
                                  handle.get_stream())
          : static_cast<vertex_t>((*nzd_unvisited_vertices).size());

      if ((next_aggregate_vertex_frontier_size * direction_optimizing_beta <
           aggregate_nzd_unvisted_vertices) &&
          (next_aggregate_vertex_frontier_size < cur_aggregate_vertex_frontier_size)) {
        top_down = true;
      }
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup6 = std::chrono::steady_clock::now();
#endif

      if (top_down) {  // swithcing to top-down
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
            handle, std::move(new_frontier_vertex_buffer));
      } else {  // staying in bottom-up
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
            handle,
            raft::device_span<vertex_t const>((*nzd_unvisited_vertices).data(),
                                              (*nzd_unvisited_vertices).size()));
      }
#if BFS_PERFORMANCE_MEASUREMENT  // FIXME: delete
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto bottomup7                     = std::chrono::steady_clock::now();
      std::chrono::duration<double> dur0 = bottomup1 - bottomup0;
      std::chrono::duration<double> dur1 = bottomup2 - bottomup1;
      std::chrono::duration<double> dur2 = bottomup3 - bottomup2;
      std::chrono::duration<double> dur3 = bottomup4 - bottomup3;
      std::chrono::duration<double> dur4 = bottomup5 - bottomup4;
      std::chrono::duration<double> dur5 = bottomup6 - bottomup5;
      std::chrono::duration<double> dur6 = bottomup7 - bottomup6;
      std::chrono::duration<double> dur  = bottomup7 - bottomup0;
      std::cout << "bottomup took " << dur.count() << " (" << dur0.count() << "," << dur1.count()
                << "," << dur2.count() << "," << dur3.count() << "," << dur4.count() << ","
                << dur5.count() << "," << dur6.count() << ") s." << std::endl;
#endif
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

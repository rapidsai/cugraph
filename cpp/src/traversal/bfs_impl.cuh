/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#include "prims/per_v_transform_reduce_if_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace {

template <typename vertex_t, typename edge_t>
struct direction_optimizing_info_t {
  rmm::device_uvector<edge_t>
    approx_out_degrees;  // if graph_view.local_vertex_partition_segment_offsets().has_value() is
                         // true, holds approximate degrees only for the high and mid degree
                         // segments; otherwise, exact
  rmm::device_uvector<uint32_t> visited_bitmap;
  std::optional<rmm::device_uvector<vertex_t>> nzd_unvisited_vertices{
    std::nullopt};  // valid only during bottom-up iterations
  std::optional<vertex_t> num_nzd_unvisited_low_degree_vertices{
    std::nullopt};  // to decide between topdown vs bottomup, relevant only when
                    // graph_view.local_vertex_partition_segment_offsets().has_value() is true
  std::optional<vertex_t> num_nzd_unvisited_hypersparse_vertices{
    std::nullopt};  // to decide between topdown vs bottomup, relevant only when
                    // graph_view.local_vertex_partition_segment_offsets().has_value() &&
                    // graph_view.use_dcs() are both true
};

template <typename vertex_t>
struct topdown_e_op_t {
  __device__ vertex_t operator()(vertex_t src,
                                 vertex_t dst,
                                 cuda::std::nullopt_t,
                                 cuda::std::nullopt_t,
                                 cuda::std::nullopt_t) const
  {
    return src;
  }
};

template <typename vertex_t, bool multi_gpu>
struct topdown_pred_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>
    prev_visited_flags{};  // visited in the previous iterations, to reduce the number of atomic
                           // operations
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool> visited_flags{};
  vertex_t dst_first{};

  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t) const
  {
    auto dst_offset = dst - dst_first;
    auto old        = prev_visited_flags.get(dst_offset);
    if (!old) { old = visited_flags.atomic_or(dst_offset, true); }
    return !old;  // haven't been visited yet.
  }
};

template <typename vertex_t>
struct bottomup_e_op_t {
  __device__ vertex_t operator()(vertex_t src,
                                 vertex_t dst,
                                 cuda::std::nullopt_t,
                                 cuda::std::nullopt_t,
                                 cuda::std::nullopt_t) const
  {
    return dst;
  }
};

template <typename vertex_t, bool multi_gpu>
struct bottomup_pred_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t const*, bool>
    prev_visited_flags{};  // visited in the previous iterations
  vertex_t dst_first{};

  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t) const
  {
    return prev_visited_flags.get(dst - dst_first);
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
                                                          raft::comms::op_t::MIN,
                                                          handle.get_stream()));
    }
    CUGRAPH_EXPECTS(
      is_sorted,
      "Invalid input arguments: input sources should be sorted in the non-descending order.");

    bool no_duplicates = (static_cast<size_t>(thrust::count_if(
                            handle.get_thrust_policy(),
                            thrust::make_counting_iterator(size_t{0}),
                            thrust::make_counting_iterator(n_sources),
                            is_first_in_run_t<decltype(sources)>{sources})) == n_sources);
    if constexpr (GraphViewType::is_multi_gpu) {
      no_duplicates = static_cast<bool>(host_scalar_allreduce(handle.get_comms(),
                                                              static_cast<int32_t>(no_duplicates),
                                                              raft::comms::op_t::MIN,
                                                              handle.get_stream()));
    }
    CUGRAPH_EXPECTS(no_duplicates,
                    "Invalid input arguments: input sources should not have duplicates.");

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

  auto segment_offsets = graph_view.local_vertex_partition_segment_offsets();

  double direction_optimizing_alpha =
    (graph_view.number_of_vertices() > 0)
      ? ((static_cast<double>(graph_view.compute_number_of_edges(handle)) /
          static_cast<double>(graph_view.number_of_vertices())) *
         (1.0 / 3.75) /* tuning parametger */)
      : double{1.0};
  constexpr vertex_t direction_optimizing_beta = 24;  // tuning parameter

  std::optional<direction_optimizing_info_t<vertex_t, edge_t>> aux_info{std::nullopt};
  if (direction_optimizing) {
    rmm::device_uvector<vertex_t> approx_out_degrees(0, handle.get_stream());
    if (segment_offsets) {  // exploit internal knowedge for exhaustive performance optimization for
                            // large-scale benchmarking (the else path is sufficient for small
                            // clusters with few tens of GPUs)
      size_t partition_idx{0};
      size_t partition_size{1};
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        auto const minor_comm_size = minor_comm.get_size();
        partition_idx              = static_cast<size_t>(minor_comm_rank);
        partition_size             = static_cast<size_t>(minor_comm_size);
      }

      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(partition_idx));
      auto edge_mask_view = graph_view.edge_mask_view();
      auto edge_partition_e_mask =
        edge_mask_view
          ? cuda::std::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
              *edge_mask_view, partition_idx)
          : cuda::std::nullopt;
      auto high_and_mid_degree_segment_size =
        (*segment_offsets)[2];  // compute local degrees for high & mid degree segments only, for
                                // low & hypersparse segments, use low_degree_threshold *
                                // partition_size * 0.5 & partition_size *
                                // hypersparse_threshold_ratio * 0.5 as approximate out degrees
      if (edge_partition_e_mask) {
        approx_out_degrees = edge_partition.compute_local_degrees_with_mask(
          (*edge_partition_e_mask).value_first(),
          thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
          thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()) +
            high_and_mid_degree_segment_size,
          handle.get_stream());
      } else {
        approx_out_degrees = edge_partition.compute_local_degrees(
          thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
          thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()) +
            high_and_mid_degree_segment_size,
          handle.get_stream());
      }
      thrust::transform(handle.get_thrust_policy(),
                        approx_out_degrees.begin(),
                        approx_out_degrees.end(),
                        approx_out_degrees.begin(),
                        multiplier_t<edge_t>{static_cast<edge_t>(
                          partition_size)});  // local_degrees => approximate global degrees
    } else {
      approx_out_degrees = graph_view.compute_out_degrees(handle);  // exact
    }

    rmm::device_uvector<uint32_t> visited_bitmap(
      packed_bool_size(graph_view.local_vertex_partition_range_size()), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 visited_bitmap.begin(),
                 visited_bitmap.end(),
                 packed_bool_empty_mask());
    thrust::for_each(
      handle.get_thrust_policy(),
      sources,
      sources + n_sources,
      [bitmap  = raft::device_span<uint32_t>(visited_bitmap.data(), visited_bitmap.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset = v - v_first;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
          bitmap[packed_bool_offset(v_offset)]);
        word.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
      });

    std::optional<vertex_t> num_nzd_unvisited_low_degree_vertices{std::nullopt};
    std::optional<vertex_t> num_nzd_unvisited_hypersparse_vertices{std::nullopt};
    if (segment_offsets) {
      num_nzd_unvisited_low_degree_vertices = (*segment_offsets)[3] - (*segment_offsets)[2];
      if (graph_view.use_dcs()) {
        num_nzd_unvisited_hypersparse_vertices = (*segment_offsets)[4] - (*segment_offsets)[3];
      }
      if (n_sources > 0) {
        std::vector<vertex_t> h_sources(n_sources);
        raft::update_host(h_sources.data(), sources, n_sources, handle.get_stream());
        handle.sync_stream();
        for (size_t i = 0; i < h_sources.size(); ++i) {
          auto v_offset = h_sources[i] - graph_view.local_vertex_partition_range_first();
          if ((v_offset >= (*segment_offsets)[2]) && (v_offset < (*segment_offsets)[3])) {
            --(*num_nzd_unvisited_low_degree_vertices);
          } else if (graph_view.use_dcs()) {
            if ((v_offset >= (*segment_offsets)[3]) && (v_offset < (*segment_offsets)[4])) {
              --(*num_nzd_unvisited_hypersparse_vertices);
            }
          }
        }
      }
    }

    aux_info =
      direction_optimizing_info_t<vertex_t, edge_t>{std::move(approx_out_degrees),
                                                    std::move(visited_bitmap),
                                                    std::nullopt,
                                                    num_nzd_unvisited_low_degree_vertices,
                                                    num_nzd_unvisited_hypersparse_vertices};
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

  // 4. BFS iteration
  vertex_t depth{0};
  bool topdown = true;
  auto cur_aggregate_frontier_size =
    static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_cur).aggregate_size());
  while (true) {
    vertex_t next_aggregate_frontier_size{};
    if (topdown) {
      topdown_pred_op_t<vertex_t, GraphViewType::is_multi_gpu> pred_op{};
      pred_op.prev_visited_flags =
        detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
          prev_dst_visited_flags.mutable_view());
      pred_op.visited_flags =
        detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t*, bool>(
          dst_visited_flags.mutable_view());
      pred_op.dst_first = graph_view.local_edge_partition_dst_range_first();

      auto [new_frontier_vertex_buffer, predecessor_buffer] =
        cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
          handle,
          graph_view,
          vertex_frontier.bucket(bucket_idx_cur),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          topdown_e_op_t<vertex_t>{},
          reduce_op::any<vertex_t>(),
          pred_op);

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

      next_aggregate_frontier_size =
        static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_next).aggregate_size());
      if (next_aggregate_frontier_size == 0) { break; }

      fill_edge_dst_property(handle,
                             graph_view,
                             vertex_frontier.bucket(bucket_idx_next).begin(),
                             vertex_frontier.bucket(bucket_idx_next).end(),
                             prev_dst_visited_flags.mutable_view(),
                             true);

      if (direction_optimizing) {
        if (vertex_frontier.bucket(bucket_idx_next).size() > 0) {
          thrust::for_each(
            handle.get_thrust_policy(),
            vertex_frontier.bucket(bucket_idx_next).begin(),
            vertex_frontier.bucket(bucket_idx_next).end(),
            [bitmap  = raft::device_span<uint32_t>((*aux_info).visited_bitmap.data(),
                                                  (*aux_info).visited_bitmap.size()),
             v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
              auto v_offset = v - v_first;
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
                bitmap[packed_bool_offset(v_offset)]);
              word.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
            });
        }

        double m_f{0.0};
        double m_u{0.0};
        {
          size_t partition_size{1};
          if constexpr (GraphViewType::is_multi_gpu) {
            auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();
            partition_size             = static_cast<size_t>(minor_comm_size);
          }

          auto f_vertex_first = vertex_frontier.bucket(bucket_idx_next).begin();
          auto f_vertex_last  = vertex_frontier.bucket(bucket_idx_next).end();

          if (segment_offsets) {
            // FIXME: this actually over-estimates for graphs with power-law degree distribution
            auto approx_low_segment_degree =
              static_cast<double>(low_degree_threshold * partition_size) * 0.5;
            auto approx_hypersparse_segment_degree =
              static_cast<double>(partition_size) * hypersparse_threshold_ratio * 0.5;
            auto f_segment_offsets = compute_key_segment_offsets(
              vertex_frontier.bucket(bucket_idx_next).begin(),
              vertex_frontier.bucket(bucket_idx_next).end(),
              raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
              graph_view.local_vertex_partition_range_first(),
              handle.get_stream());
            *((*aux_info).num_nzd_unvisited_low_degree_vertices) -=
              (f_segment_offsets[3] - f_segment_offsets[2]);
            if (graph_view.use_dcs()) {
              *((*aux_info).num_nzd_unvisited_hypersparse_vertices) -=
                (f_segment_offsets[4] - f_segment_offsets[3]);
            }
            f_vertex_last = f_vertex_first + f_segment_offsets[2];
            m_f           = static_cast<double>((f_segment_offsets[3] - f_segment_offsets[2])) *
                  approx_low_segment_degree;
            if (graph_view.use_dcs()) {
              m_f += static_cast<double>(f_segment_offsets[4] - f_segment_offsets[3]) *
                     approx_hypersparse_segment_degree;
            }

            m_u = static_cast<double>(*((*aux_info).num_nzd_unvisited_low_degree_vertices)) *
                  approx_low_segment_degree;
            if (graph_view.use_dcs()) {
              m_u += static_cast<double>(*((*aux_info).num_nzd_unvisited_hypersparse_vertices)) *
                     approx_hypersparse_segment_degree;
            }
          }

          m_f += static_cast<double>(thrust::transform_reduce(
            handle.get_thrust_policy(),
            f_vertex_first,
            f_vertex_last,
            cuda::proclaim_return_type<edge_t>(
              [out_degrees = raft::device_span<edge_t const>((*aux_info).approx_out_degrees.data(),
                                                             (*aux_info).approx_out_degrees.size()),
               v_first = graph_view.local_vertex_partition_range_first()] __device__(vertex_t v) {
                auto v_offset = v - v_first;
                return out_degrees[v_offset];
              }),
            edge_t{0},
            thrust::plus<edge_t>{}));

          m_u += static_cast<double>(thrust::transform_reduce(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(vertex_t{0}),
            thrust::make_counting_iterator(segment_offsets
                                             ? (*segment_offsets)[2]
                                             : graph_view.local_vertex_partition_range_size()),
            cuda::proclaim_return_type<edge_t>(
              [out_degrees = raft::device_span<edge_t const>((*aux_info).approx_out_degrees.data(),
                                                             (*aux_info).approx_out_degrees.size()),
               bitmap      = raft::device_span<uint32_t const>(
                 (*aux_info).visited_bitmap.data(),
                 (*aux_info).visited_bitmap.size())] __device__(vertex_t v_offset) {
                auto word = bitmap[packed_bool_offset(v_offset)];
                if ((word & packed_bool_mask(v_offset)) != packed_bool_empty_mask()) {  // visited
                  return edge_t{0};
                } else {
                  return out_degrees[v_offset];
                }
              }),
            edge_t{0},
            thrust::plus<edge_t>{}));
        }

        auto aggregate_m_f = m_f;
        auto aggregate_m_u = m_u;
        if constexpr (GraphViewType::is_multi_gpu) {
          auto tmp      = host_scalar_allreduce(handle.get_comms(),
                                           thrust::make_tuple(m_f, m_u),
                                           raft::comms::op_t::SUM,
                                           handle.get_stream());
          aggregate_m_f = thrust::get<0>(tmp);
          aggregate_m_u = thrust::get<1>(tmp);
        }
        if ((aggregate_m_f * direction_optimizing_alpha > aggregate_m_u) &&
            (next_aggregate_frontier_size >= cur_aggregate_frontier_size)) {
          topdown                            = false;
          (*aux_info).nzd_unvisited_vertices = rmm::device_uvector<vertex_t>(
            segment_offsets ? *((*segment_offsets).rbegin() + 1)
                            : graph_view.local_vertex_partition_range_size(),
            handle.get_stream());
          (*((*aux_info).nzd_unvisited_vertices))
            .resize(
              thrust::distance(
                (*((*aux_info).nzd_unvisited_vertices)).begin(),
                thrust::copy_if(
                  handle.get_thrust_policy(),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
                  thrust::make_counting_iterator(
                    segment_offsets ? graph_view.local_vertex_partition_range_first() +
                                        *((*segment_offsets).rbegin() + 1)
                                    : graph_view.local_vertex_partition_range_last()),
                  (*((*aux_info).nzd_unvisited_vertices)).begin(),
                  [bitmap  = raft::device_span<uint32_t const>((*aux_info).visited_bitmap.data(),
                                                              (*aux_info).visited_bitmap.size()),
                   v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                    auto v_offset = v - v_first;
                    auto word     = bitmap[packed_bool_offset(v_offset)];
                    return ((word & packed_bool_mask(v_offset)) == packed_bool_empty_mask());
                  })),
              handle.get_stream());
        }
      }

      if (topdown) {  // staying in top-down
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(handle);
        vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
      } else {  // swithcing to bottom-up
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
            handle,
            raft::device_span<vertex_t const>((*((*aux_info).nzd_unvisited_vertices)).data(),
                                              (*((*aux_info).nzd_unvisited_vertices)).size()));
        vertex_frontier.bucket(bucket_idx_next) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(handle);
      }
    } else {  // bottom up
      rmm::device_uvector<vertex_t> new_frontier_vertex_buffer(0, handle.get_stream());
      {
        bottomup_e_op_t<vertex_t> e_op{};
        bottomup_pred_op_t<vertex_t, GraphViewType::is_multi_gpu> pred_op{};
        pred_op.prev_visited_flags =
          detail::edge_partition_endpoint_property_device_view_t<vertex_t, uint32_t const*, bool>(
            prev_dst_visited_flags.view());
        pred_op.dst_first = graph_view.local_edge_partition_dst_range_first();

        rmm::device_uvector<vertex_t> predecessor_buffer(
          vertex_frontier.bucket(bucket_idx_cur).size(), handle.get_stream());
        per_v_transform_reduce_if_outgoing_e(handle,
                                             graph_view,
                                             vertex_frontier.bucket(bucket_idx_cur),
                                             edge_src_dummy_property_t{}.view(),
                                             edge_dst_dummy_property_t{}.view(),
                                             edge_dummy_property_t{}.view(),
                                             e_op,
                                             invalid_vertex,
                                             reduce_op::any<vertex_t>(),
                                             pred_op,
                                             predecessor_buffer.begin(),
                                             true);
        auto input_pair_first = thrust::make_zip_iterator(thrust::make_constant_iterator(depth + 1),
                                                          predecessor_buffer.begin());

        // FIXME: this scatter_if and the resize below can be concurrently executed.
        thrust::scatter_if(
          handle.get_thrust_policy(),
          input_pair_first,
          input_pair_first + predecessor_buffer.size(),
          thrust::make_transform_iterator(
            vertex_frontier.bucket(bucket_idx_cur).cbegin(),
            detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
          predecessor_buffer.begin(),
          thrust::make_zip_iterator(distances, predecessor_first),
          detail::is_not_equal_t<vertex_t>{invalid_vertex});

        new_frontier_vertex_buffer.resize(predecessor_buffer.size(), handle.get_stream());
        new_frontier_vertex_buffer.resize(
          thrust::distance(new_frontier_vertex_buffer.begin(),
                           thrust::copy_if(handle.get_thrust_policy(),
                                           vertex_frontier.bucket(bucket_idx_cur).cbegin(),
                                           vertex_frontier.bucket(bucket_idx_cur).cend(),
                                           predecessor_buffer.begin(),
                                           new_frontier_vertex_buffer.begin(),
                                           detail::is_not_equal_t<vertex_t>{invalid_vertex})),
          handle.get_stream());

        assert(direction_optimizing);

        thrust::for_each(
          handle.get_thrust_policy(),
          new_frontier_vertex_buffer.begin(),
          new_frontier_vertex_buffer.end(),
          [bitmap  = raft::device_span<uint32_t>((*aux_info).visited_bitmap.data(),
                                                (*aux_info).visited_bitmap.size()),
           v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
            auto v_offset = v - v_first;
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
              bitmap[packed_bool_offset(v_offset)]);
            word.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
          });
        (*((*aux_info).nzd_unvisited_vertices))
          .resize(
            thrust::distance(
              (*((*aux_info).nzd_unvisited_vertices)).begin(),
              thrust::remove_if(
                handle.get_thrust_policy(),
                (*((*aux_info).nzd_unvisited_vertices)).begin(),
                (*((*aux_info).nzd_unvisited_vertices)).end(),
                [bitmap  = raft::device_span<uint32_t const>((*aux_info).visited_bitmap.data(),
                                                            (*aux_info).visited_bitmap.size()),
                 v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                  auto v_offset = v - v_first;
                  auto word     = bitmap[packed_bool_offset(v_offset)];
                  return ((word & packed_bool_mask(v_offset)) != packed_bool_empty_mask());
                })),
            handle.get_stream());

        if (segment_offsets) {
          auto key_segment_offsets = compute_key_segment_offsets(
            new_frontier_vertex_buffer.begin(),
            new_frontier_vertex_buffer.end(),
            raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
            graph_view.local_vertex_partition_range_first(),
            handle.get_stream());
          *((*aux_info).num_nzd_unvisited_low_degree_vertices) -=
            key_segment_offsets[3] - key_segment_offsets[2];
          if (graph_view.use_dcs()) {
            *((*aux_info).num_nzd_unvisited_hypersparse_vertices) -=
              key_segment_offsets[4] - key_segment_offsets[3];
          }
        }
      }

      next_aggregate_frontier_size = static_cast<vertex_t>(new_frontier_vertex_buffer.size());
      auto aggregate_nzd_unvisited_vertices =
        static_cast<vertex_t>((*((*aux_info).nzd_unvisited_vertices)).size());
      if constexpr (GraphViewType::is_multi_gpu) {
        auto tmp = host_scalar_allreduce(
          handle.get_comms(),
          thrust::make_tuple(next_aggregate_frontier_size, aggregate_nzd_unvisited_vertices),
          raft::comms::op_t::SUM,
          handle.get_stream());
        next_aggregate_frontier_size     = thrust::get<0>(tmp);
        aggregate_nzd_unvisited_vertices = thrust::get<1>(tmp);
      }

      if (next_aggregate_frontier_size == 0) { break; }

      fill_edge_dst_property(handle,
                             graph_view,
                             new_frontier_vertex_buffer.begin(),
                             new_frontier_vertex_buffer.end(),
                             prev_dst_visited_flags.mutable_view(),
                             true);

      if ((next_aggregate_frontier_size * direction_optimizing_beta <
           aggregate_nzd_unvisited_vertices) &&
          (next_aggregate_frontier_size < cur_aggregate_frontier_size)) {
        topdown = true;
      }

      if (topdown) {  // swithcing to top-down
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
            handle, std::move(new_frontier_vertex_buffer));
      } else {  // staying in bottom-up
        vertex_frontier.bucket(bucket_idx_cur) =
          key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true>(
            handle,
            raft::device_span<vertex_t const>((*((*aux_info).nzd_unvisited_vertices)).data(),
                                              ((*(*aux_info).nzd_unvisited_vertices)).size()));
      }
    }
    cur_aggregate_frontier_size = next_aggregate_frontier_size;

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

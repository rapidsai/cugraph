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

#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/reduce_v.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstddef>

namespace cugraph {

namespace {

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename edge_t>
struct v_to_core_number_t {
  edge_t const* core_numbers{nullptr};
  vertex_t v_first{0};

  __device__ edge_t operator()(vertex_t v) const { return core_numbers[v - v_first]; }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename edge_t>
struct mult_degree_by_two_t {
  __device__ edge_t operator()(edge_t d) const { return d * edge_t{2}; }
};

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
void core_number(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                 edge_t* core_numbers,
                 k_core_degree_type_t degree_type,
                 size_t k_first,
                 size_t k_last,
                 bool do_expensive_check)
{
  // check input arguments.

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input argument: core_number currently supports only undirected graphs.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input argument: core_number currently does not support multi-graphs.");
  CUGRAPH_EXPECTS((degree_type == k_core_degree_type_t::IN) ||
                    (degree_type == k_core_degree_type_t::OUT) ||
                    (degree_type == k_core_degree_type_t::INOUT),
                  "Invalid input argument: degree_type should be IN, OUT, or INOUT.");
  CUGRAPH_EXPECTS(k_first <= k_last, "Invalid input argument: k_first <= k_last.");

  // Exclude self-loops

  std::optional<cugraph::edge_property_t<edge_t, bool>> self_loop_edge_mask{std::nullopt};
  auto cur_graph_view = graph_view;
  if (cur_graph_view.count_self_loops(handle) > edge_t{0}) {
    self_loop_edge_mask = cugraph::edge_property_t<edge_t, bool>(handle, cur_graph_view);
    if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
    cugraph::fill_edge_property(handle, cur_graph_view, self_loop_edge_mask->mutable_view(), false);

    transform_e(handle,
                graph_view,
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_dummy_property_t{}.view(),
                cuda::proclaim_return_type<bool>(
                  [] __device__(auto src, auto dst, auto, auto, auto) { return src != dst; }),
                self_loop_edge_mask->mutable_view());

    cur_graph_view.attach_edge_mask(self_loop_edge_mask->view());
  }

  // initialize core_numbers to degrees (upper-bound)

  if (cur_graph_view.is_symmetric()) {  // in-degree == out-degree
    auto out_degrees = cur_graph_view.compute_out_degrees(handle);
    if ((degree_type == k_core_degree_type_t::IN) || (degree_type == k_core_degree_type_t::OUT)) {
      thrust::copy(
        handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), core_numbers);
    } else {
      auto inout_degree_first =
        thrust::make_transform_iterator(out_degrees.begin(), mult_degree_by_two_t<edge_t>{});
      thrust::copy(handle.get_thrust_policy(),
                   inout_degree_first,
                   inout_degree_first + out_degrees.size(),
                   core_numbers);
    }
  } else {
    if (degree_type == k_core_degree_type_t::IN) {
      auto in_degrees = cur_graph_view.compute_in_degrees(handle);
      thrust::copy(handle.get_thrust_policy(), in_degrees.begin(), in_degrees.end(), core_numbers);
    } else if (degree_type == k_core_degree_type_t::OUT) {
      auto out_degrees = cur_graph_view.compute_out_degrees(handle);
      thrust::copy(
        handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), core_numbers);
    } else {
      auto in_degrees  = cur_graph_view.compute_in_degrees(handle);
      auto out_degrees = cur_graph_view.compute_out_degrees(handle);
      auto degree_pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(in_degrees.begin(), out_degrees.begin()));
      thrust::transform(handle.get_thrust_policy(),
                        degree_pair_first,
                        degree_pair_first + in_degrees.size(),
                        core_numbers,
                        [] __device__(auto p) { return thrust::get<0>(p) + thrust::get<1>(p); });
    }
  }

  // remove 0 degree vertices (as they already belong to 0-core and they don't affect core numbers)
  // and clip core numbers of the "less than k_first degree" vertices to 0

  rmm::device_uvector<vertex_t> remaining_vertices(
    cur_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  remaining_vertices.resize(
    cuda::std::distance(
      remaining_vertices.begin(),
      thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(cur_graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(cur_graph_view.local_vertex_partition_range_last()),
        remaining_vertices.begin(),
        [core_numbers, v_first = cur_graph_view.local_vertex_partition_range_first()] __device__(
          auto v) { return core_numbers[v - v_first] > edge_t{0}; })),
    handle.get_stream());

  if (k_first > 1) {
    thrust::for_each(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [k_first,
       core_numbers,
       v_first = cur_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        if (core_numbers[v - v_first] < k_first) { core_numbers[v - v_first] = edge_t{0}; }
      });
  }

  // start iteration

  constexpr size_t bucket_idx_cur  = 0;
  constexpr size_t bucket_idx_next = 1;
  constexpr size_t num_buckets     = 2;

  vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, num_buckets);

  edge_dst_property_t<edge_t, bool> edge_dst_valids(handle, cur_graph_view);
  fill_edge_dst_property(handle, cur_graph_view, edge_dst_valids.mutable_view(), true);
  if (!cur_graph_view.is_symmetric() &&
      degree_type != k_core_degree_type_t::INOUT) {  // 0 core number vertex may have non-zero
                                                     // in|out-degrees (so still can be accessed)
    rmm::device_uvector<vertex_t> zero_degree_vertices(
      cur_graph_view.local_vertex_partition_range_size() - remaining_vertices.size(),
      handle.get_stream());
    thrust::copy_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(cur_graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(cur_graph_view.local_vertex_partition_range_last()),
      zero_degree_vertices.begin(),
      [core_numbers, v_first = cur_graph_view.local_vertex_partition_range_first()] __device__(
        auto v) { return core_numbers[v - v_first] == edge_t{0}; });
    fill_edge_dst_property(handle,
                           cur_graph_view,
                           zero_degree_vertices.begin(),
                           zero_degree_vertices.end(),
                           edge_dst_valids.mutable_view(),
                           false);
  } else {  // skip invaliding 0 core number vertices as they won't be accessed anyways
    /* nothing to do */
  }

  auto k = std::max(k_first, size_t{2});  // degree 0|1 vertices belong to 0|1-core
  if (cur_graph_view.is_symmetric() && (degree_type == k_core_degree_type_t::INOUT) &&
      ((k % 2) == 1)) {  // core numbers are always even numbers if symmetric and INOUT
    ++k;
  }
  while (k <= k_last) {
    size_t aggregate_num_remaining_vertices{0};
    if constexpr (multi_gpu) {
      auto& comm                       = handle.get_comms();
      aggregate_num_remaining_vertices = host_scalar_allreduce(
        comm, remaining_vertices.size(), raft::comms::op_t::SUM, handle.get_stream());
    } else {
      aggregate_num_remaining_vertices = remaining_vertices.size();
    }
    if (aggregate_num_remaining_vertices == 0) { break; }

    // FIXME: scanning the remaining vertices can add significant overhead if the number of distinct
    // core numbers in [k_first, std::min(max_degree, k_last)] is large and there are many high core
    // number vertices (so the number of remaining vertices remains large for many iterations). Need
    // more tuning (e.g. Possibly use a logarithmic binning) if we encounter such use cases.
    auto less_than_k_first = thrust::stable_partition(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [core_numbers, k, v_first = cur_graph_view.local_vertex_partition_range_first()] __device__(
        auto v) { return core_numbers[v - v_first] >= k; });
    vertex_frontier.bucket(bucket_idx_cur).insert(less_than_k_first, remaining_vertices.end());
    remaining_vertices.resize(cuda::std::distance(remaining_vertices.begin(), less_than_k_first),
                              handle.get_stream());

    auto delta = (cur_graph_view.is_symmetric() && (degree_type == k_core_degree_type_t::INOUT))
                   ? edge_t{2}
                   : edge_t{1};
    if (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() > 0) {
      do {
        fill_edge_dst_property(handle,
                               cur_graph_view,
                               vertex_frontier.bucket(bucket_idx_cur).begin(),
                               vertex_frontier.bucket(bucket_idx_cur).end(),
                               edge_dst_valids.mutable_view(),
                               false);

        // FIXME: If most vertices have core numbers less than k, (dst_val >= k) will be mostly
        // false leading to too many unnecessary edge traversals (this is especially problematic if
        // the number of distinct core numbers in [k_first, std::min(max_degree, k_last)] is large).
        // There are two potential solutions: 1) extract a sub-graph and work on the sub-graph & 2)
        // mask-out/delete edges.
        if (cur_graph_view.is_symmetric() ||
            ((degree_type == k_core_degree_type_t::IN) ||
             (degree_type == k_core_degree_type_t::INOUT))) {  // decrement in-degrees
          auto [new_frontier_vertex_buffer, delta_buffer] =
            cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
              handle,
              cur_graph_view,
              vertex_frontier.bucket(bucket_idx_cur),
              edge_src_dummy_property_t{}.view(),
              edge_dst_valids.view(),
              edge_dummy_property_t{}.view(),
              cuda::proclaim_return_type<edge_t>(
                [k, delta] __device__(auto, auto, auto, auto, auto) { return delta; }),
              reduce_op::plus<edge_t>(),
              cuda::proclaim_return_type<bool>(
                [k, delta] __device__(auto, auto, auto, bool dst_valid, auto) {
                  return dst_valid;
                }));

          update_v_frontier(
            handle,
            cur_graph_view,
            std::move(new_frontier_vertex_buffer),
            std::move(delta_buffer),
            vertex_frontier,
            raft::host_span<size_t const>(&bucket_idx_next, size_t{1}),
            core_numbers,
            core_numbers,
            [k_first,
             k,
             delta,
             v_first =
               cur_graph_view.local_vertex_partition_range_first()] __device__(auto v,
                                                                               auto v_val,
                                                                               auto pushed_val) {
              auto new_core_number = v_val >= pushed_val ? v_val - pushed_val : edge_t{0};
              new_core_number      = new_core_number < (k - delta) ? (k - delta) : new_core_number;
              new_core_number      = new_core_number < k_first ? edge_t{0} : new_core_number;
              return thrust::make_tuple(cuda::std::optional<size_t>{bucket_idx_next},
                                        cuda::std::optional<edge_t>{new_core_number});
            });
        }

        if (!cur_graph_view.is_symmetric() &&
            ((degree_type == k_core_degree_type_t::OUT) ||
             (degree_type == k_core_degree_type_t::INOUT))) {  // decrement out-degrees
          // FIXME: we can create a transposed copy of the input graph (note that currently,
          // transpose works only on graph_t (and does not work on graph_view_t)).
          CUGRAPH_FAIL("unimplemented.");
        }

        vertex_frontier.bucket(bucket_idx_next)
          .resize(static_cast<size_t>(cuda::std::distance(
            vertex_frontier.bucket(bucket_idx_next).begin(),
            thrust::remove_if(
              handle.get_thrust_policy(),
              vertex_frontier.bucket(bucket_idx_next).begin(),
              vertex_frontier.bucket(bucket_idx_next).end(),
              [core_numbers,
               k,
               v_first = cur_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                return core_numbers[v - v_first] >= k;
              }))));
        vertex_frontier.bucket(bucket_idx_next).shrink_to_fit();

        vertex_frontier.bucket(bucket_idx_cur).clear();
        vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
        vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
      } while (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() > 0);

      // FIXME: scanning the remaining vertices can add significant overhead if the number of
      // distinct core numbers in [k_first, std::min(max_degree, k_last)] is large and there are
      // many high core number vertices (so the number of remaining vertices remains large for many
      // iterations). Need more tuning (e.g. Possibly use a logarithmic binning) if we encounter
      // such use cases.
      remaining_vertices.resize(
        cuda::std::distance(
          remaining_vertices.begin(),
          thrust::remove_if(
            handle.get_thrust_policy(),
            remaining_vertices.begin(),
            remaining_vertices.end(),
            [core_numbers,
             k,
             v_first = cur_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
              return core_numbers[v - v_first] < k;
            })),
        handle.get_stream());
      k += delta;
    } else {
      auto remaining_vertex_core_number_first = thrust::make_transform_iterator(
        remaining_vertices.begin(),
        v_to_core_number_t<vertex_t, edge_t>{core_numbers,
                                             cur_graph_view.local_vertex_partition_range_first()});
      auto min_core_number =
        thrust::reduce(handle.get_thrust_policy(),
                       remaining_vertex_core_number_first,
                       remaining_vertex_core_number_first + remaining_vertices.size(),
                       std::numeric_limits<edge_t>::max(),
                       thrust::minimum<edge_t>{});
      if constexpr (multi_gpu) {
        min_core_number = host_scalar_allreduce(
          handle.get_comms(), min_core_number, raft::comms::op_t::MIN, handle.get_stream());
      }
      k = std::max(k + delta, static_cast<size_t>(min_core_number + edge_t{delta}));
    }
  }
}

}  // namespace cugraph

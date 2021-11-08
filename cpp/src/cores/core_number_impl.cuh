/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cugraph/prims/copy_to_adj_matrix_row_col.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/row_col_properties.cuh>
#include <cugraph/prims/update_frontier_v_push_if_out_nbr.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>

#include <cstddef>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void core_number(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                 edge_t* core_numbers,
                 k_core_degree_type_t degree_type,
                 size_t k_first,
                 size_t k_last,
                 bool do_expensive_check)
{
  // check input arguments.

  CUGRAPH_EXPECTS((degree_type == k_core_degree_type_t::IN) ||
                    (degree_type == k_core_degree_type_t::OUT) ||
                    (degree_type == k_core_degree_type_t::INOUT),
                  "Invalid input argument: degree_type should be IN, OUT, or INOUT.");
  CUGRAPH_EXPECTS(k_first <= k_last, "Invalid input argument: k_first <= k_last.");

  if (do_expensive_check) {
    if (graph_view.is_multigraph()) {
      CUGRAPH_FAIL("unimplemented.");  // check for multi-edges
    }
    CUGRAPH_FAIL("unimplemented.");  // check for self-loops
  }

  // initialize core_numbers to degrees

  if (graph_view.is_symmetric()) {  // in-degree == out-degree
    auto out_degrees = graph_view.compute_out_degrees(handle);
    if ((degree_type == k_core_degree_type_t::IN) || (degree_type == k_core_degree_type_t::OUT)) {
      thrust::copy(
        handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), core_numbers);
    } else {
      auto inout_degree_first = thrust::make_transform_iterator(
        out_degrees.begin(), [] __device__(auto d) { return d * edge_t{2}; });
      thrust::copy(handle.get_thrust_policy(),
                   inout_degree_first,
                   inout_degree_first + out_degrees.size(),
                   core_numbers);
    }
  } else {
    if (degree_type == k_core_degree_type_t::IN) {
      auto in_degrees = graph_view.compute_in_degrees(handle);
      thrust::copy(handle.get_thrust_policy(), in_degrees.begin(), in_degrees.end(), core_numbers);
    } else if (degree_type == k_core_degree_type_t::OUT) {
      auto out_degrees = graph_view.compute_out_degrees(handle);
      thrust::copy(
        handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), core_numbers);
    } else {
      auto in_degrees  = graph_view.compute_in_degrees(handle);
      auto out_degrees = graph_view.compute_out_degrees(handle);
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

  rmm::device_uvector<vertex_t> remaining_vertices(graph_view.get_number_of_local_vertices(),
                                                   handle.get_stream());
  remaining_vertices.resize(
    thrust::distance(
      remaining_vertices.begin(),
      thrust::copy_if(handle.get_thrust_policy(),
                      thrust::make_counting_iterator(graph_view.get_local_vertex_first()),
                      thrust::make_counting_iterator(graph_view.get_local_vertex_last()),
                      remaining_vertices.begin(),
                      [core_numbers, v_first = graph_view.get_local_vertex_first()] __device__(
                        auto v) { return core_numbers[v - v_first] > edge_t{0}; })),
    handle.get_stream());

  if (k_first > 1) {
    thrust::for_each(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [k_first, core_numbers, v_first = graph_view.get_local_vertex_first()] __device__(auto v) {
        if (core_numbers[v - v_first] < k_first) { core_numbers[v - v_first] = edge_t{0}; }
      });
  }

  // start iteration

  enum class Bucket { cur, next, num_buckets };
  VertexFrontier<vertex_t, void, multi_gpu, static_cast<size_t>(Bucket::num_buckets)>
    vertex_frontier(handle);

  col_properties_t<graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu>, edge_t>
    dst_core_numbers(handle, graph_view);
  copy_to_adj_matrix_col(handle, graph_view, core_numbers, dst_core_numbers);

  auto k = std::max(k_first, size_t{2});  // degree 0|1 vertices belong to 0|1-core
  if (graph_view.is_symmetric() && (degree_type == k_core_degree_type_t::INOUT) &&
      ((k % 2) == 1)) {  // core numbers are always even numbers if symmetric and INOUT
    ++k;
  }
  while (k <= k_last) {
    size_t aggregate_num_remaining_vertices{0};
    if constexpr (multi_gpu) {
      auto& comm = handle.get_comms();
      aggregate_num_remaining_vertices = host_scalar_allreduce(
        comm, remaining_vertices.size(), raft::comms::op_t::SUM, handle.get_stream());
    } else {
      aggregate_num_remaining_vertices = remaining_vertices.size();
    }
    if (aggregate_num_remaining_vertices == 0) { break; }

    // FIXME: scanning the remaining vertices can add significant overhead if std::min(max_degree,
    // k_last) >> k_first.
    auto less_than_k_first = thrust::stable_partition(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [core_numbers, k, v_first = graph_view.get_local_vertex_first()] __device__(auto v) {
        return core_numbers[v - v_first] >= k;
      });
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
      .insert(less_than_k_first, remaining_vertices.end());
    remaining_vertices.resize(thrust::distance(remaining_vertices.begin(), less_than_k_first),
                              handle.get_stream());

    auto delta = (graph_view.is_symmetric() && (degree_type == k_core_degree_type_t::INOUT))
                   ? edge_t{2}
                   : edge_t{1};
    if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() > 0) {
      do {
        // FIXME: If most vertices have core numbers less than k, (dst_val >= k) will be mostly
        // false leading to too many unnecessary edge traversals (this is especially problematic if
        // std::min(max_degree, k_last) >> k_first). There are two potential solutions: 1) extract a
        // sub-graph and work on the sub-graph & 2) mask-out/delete edges.
        if (graph_view.is_symmetric() || ((degree_type == k_core_degree_type_t::IN) ||
                                          (degree_type == k_core_degree_type_t::INOUT))) {
          update_frontier_v_push_if_out_nbr(
            handle,
            graph_view,
            vertex_frontier,
            static_cast<size_t>(Bucket::cur),
            std::vector<size_t>{static_cast<size_t>(Bucket::next)},
            dummy_properties_t<vertex_t>{}.device_view(),
            dst_core_numbers.device_view(),
            [k, delta] __device__(vertex_t src, vertex_t dst, auto, auto dst_val) {
              return dst_val >= k ? thrust::optional<edge_t>{delta} : thrust::nullopt;
            },
            reduce_op::plus<edge_t>(),
            core_numbers,
            core_numbers,
            [k, delta, v_first = graph_view.get_local_vertex_first()] __device__(
              auto, auto v_val, auto pushed_val) {
              auto old_core_number = v_val;
              auto new_core_number = old_core_number >= (pushed_val + k - delta)
                                       ? (old_core_number - pushed_val)
                                       : (k - delta);
              return thrust::optional<thrust::tuple<size_t, edge_t>>{
                thrust::make_tuple(static_cast<size_t>(Bucket::next), new_core_number)};
            });
        }

        if (!graph_view.is_symmetric() && ((degree_type == k_core_degree_type_t::OUT) ||
                                           (degree_type == k_core_degree_type_t::INOUT))) {
          // FIXME: we can create a transposed copy of the input graph.
          CUGRAPH_FAIL("unimplemented.");
        }

        copy_to_adj_matrix_col(
          handle,
          graph_view,
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).begin(),
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).end(),
          core_numbers,
          dst_core_numbers);

        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next))
          .resize(static_cast<size_t>(thrust::distance(
            vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).begin(),
            thrust::remove_if(
              handle.get_thrust_policy(),
              vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).begin(),
              vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).end(),
              [core_numbers, k, v_first = graph_view.get_local_vertex_first()] __device__(auto v) {
                return core_numbers[v - v_first] >= k;
              }))));
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).shrink_to_fit();

        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).clear();
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).shrink_to_fit();
        vertex_frontier.swap_buckets(static_cast<size_t>(Bucket::cur),
                                     static_cast<size_t>(Bucket::next));
      } while (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() > 0);

      // FIXME: scanning the remaining vertices can add significant overhead if std::min(max_degree, k_last) >> k_first.
      remaining_vertices.resize(
        thrust::distance(
          remaining_vertices.begin(),
          thrust::remove_if(
            handle.get_thrust_policy(),
            remaining_vertices.begin(),
            remaining_vertices.end(),
            [core_numbers, k, v_first = graph_view.get_local_vertex_first()] __device__(auto v) {
              return core_numbers[v - v_first] < k;
            })),
        handle.get_stream());
      k += delta;
    } else {
      auto remaining_vertex_core_number_first = thrust::make_transform_iterator(
        remaining_vertices.begin(),
        [core_numbers, v_first = graph_view.get_local_vertex_first()] __device__(auto v) {
          return core_numbers[v - v_first];
        });
      auto min_core_number =
        reduce_v(handle,
                 graph_view,
                 remaining_vertex_core_number_first,
                 remaining_vertex_core_number_first + remaining_vertices.size(),
                 std::numeric_limits<edge_t>::max(),
                 raft::comms::op_t::MIN);
      k = std::max(k, static_cast<size_t>(min_core_number + edge_t{delta}));
    }
  }

  // clip core numbers to k_last

  if (k_last < std::numeric_limits<size_t>::max()) {
    thrust::transform(handle.get_thrust_policy(),
                      core_numbers,
                      core_numbers + graph_view.get_number_of_local_vertices(),
                      core_numbers,
                      [k_last = static_cast<edge_t>(k_last), op = thrust::minimum<edge_t>{}] __device__(auto c) {
                        return op(c, k_last);
                      });
  }
}

}  // namespace cugraph

/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/distance.h>
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

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> find_forest(
  raft::handle_t const& handle, graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view, bool do_expensive_check)
{
  // check input arguments.

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input argument: find_forest currently supports only undirected graphs.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input argument: core_number currently does not support multi-graphs.");

  if (do_expensive_check) {
    CUGRAPH_EXPECTS(graph_view.count_self_loops(handle) == 0,
                    "Invalid input argument: graph_view has self-loops.");
  }

  // initilaize parents, degrees, and remaining_vertices

  rmm::device_uvector<vertex_t> parents(graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               parents.begin(),
               parents.end(),
               cugraph::invalid_vertex_id_v<vertex_t>);

  edge_dst_property_t<decltype(graph_view), bool> edge_dst_valids(handle, graph_view);
  fill_edge_dst_property(handle,
                        graph_view,
                        edge_dst_parents.mutable_view(),
                        true);

  auto degrees = graph_view.compute_out_degrees(handle);

  rmm::device_uvector<vertex_t> remaining_vertices(graph_view.local_vertex_partition_range_size(),
                                                   handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(),
                   remaining_vertices.begin(),
                   remaining_vertices.end(),
                   graph_view.local_vertex_partition_range_first());

  // recursively remove degree 0 and degree 1 vertices
  while (true) {
    auto deg0_v_first = thrust::stable_partition(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [degrees = raft::device_span<edge_t const>(degrees.data(), degrees.size()),
       v_first = graph_view.local_vertex_partition_range_first()](auto v) {
        return degrees[v - v_first] > 1;
      });
    auto deg1_v_first = thrust::stable_partition(
      handle.get_thrust_policy(),
      deg0_v_first,
      remaining_vertices.end(),
      [degrees = raft::device_span<edge_t const>(degrees.data(), degrees.size()),
       v_first = graph_view.local_vertex_partition_range_first()](auto v) {
        return degrees[v - v_first] == 0;
      });

    // mark new degree 0 & degree 1 vertices as invalid

    fill_edge_dst_property(handle,
                          graph_view,
                          deg0_v_first,
                          remaining_vertices.end(),
                          edge_dst_valids.mutable_view(),
                          false);

    // for degree 0 vertices, set parents to itself

    thrust::for_each(handle.get_thrust_policy(),
                     deg0_v_first,
                     deg1_v_first,
                     [parents = raft::device_span<vertex_t>(parents.data(), parents.size()),
                      v_first = graph_view.local_vertex_partition_range_first()](auto v) {
                       parents[v - v_first] = v;
                     });

    if (thrust::distance(deg1_v_first, remaining_vertices.end()) >
        0) {  // for degree 1 vertices, find degree 1 vertices' only neighbors. Set each degree 1
              // vertex's parent to its only neighbor. Reduce the neighbors' degree by 1.
      size_t constexpr num_buckets            = 1;
      bool constexpr sorted_unique_key_bucket = true;
      vertex_frontier_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> vertex_frontier(
        handle, num_buckets);
      vertex_frontier.bucket(0).insert(deg1_v_first, remaining_vertices.end());

      auto edges = extract_transform_if_v_frontier_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        edge_dst_valids.view(),
        edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<thrust::tuple<vertex_t, vertex_t>>(
          [] __device__(auto src, auto dst, auto, auto, auto) {
            return thrust::make_tuple(src, dst);
          }),
          cuda::proclaim_return_type<bool>([] __device__(auto, auto, auto, auto valid, auto) {
            return valid;
          }));

      // decrease the neighbors' degree by 1

      std::optional<rmm::device_uvector<vertex_t>> shuffled_edge_dsts{std::nullopt};
      if constexpr (multi_gpu) {
        rmm::device_uvector<vertex_t> edge_dsts(std::get<1>(edges).size(), handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     std::get<1>(edges).begin(),
                     std::get<1>(edges).end(),
                     edge_dsts.begin());
        shuffled_edge_dsts =
          shuffle_local_edge_dsts(handle, std::move(edge_dsts));
      }
      thrust::for_each(
        handle.get_thrust_policy(),
        shuffled_edge_dsts ? shuffled_edge_dsts->begin() : std::get<1>(edges).begin(),
        shuffled_edge_dsts ? shuffled_edge_dsts->end() : std::get<1>(edges).end(),
        [degrees = raft::device_span<edge_t const>(degrees.data(), degrees.size()),
         v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          cuda::atomic_ref<edge_t, cuda::thread_scope_device> degree(degrees[v - v_first]);
          degree.fetch_sub(1);
        });
      shuffled_edge_dsts = std::nullopt;

      // update parents

      std::optional<dataframe_buffer_type_t<thrust::tuple<vertex_t, vertex_t>>> shuffled_edges{
        std::nullopt};
      if constexpr (multi_gpu) {
        rmm::device_uvector<vertex_t> edge_srcs(std::get<0>(edges).size(), handle.get_stream());
        rmm::device_uvector<vertex_t> edge_dsts(edge_srcs.size(), handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(edges),
                     get_dataframe_buffer_end(edges),
                     thrust::make_zip_iterator(edge_srcs.begin(), edge_dsts.begin()));
        shuffled_edges = shuffle_local_edge_src_value_pairs(
          handle, std::move(edge_srcs), std::move(edge_dsts));
      }
      auto pair_first = get_dataframe_buffer_begin(shuffled_edges ? *shuffled_edges : edges);
      thrust::for_each(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + size_dataframe_buffer(shuffled_edges ? *shuffled_edges : edges),
        [parents = raft::device_span<vertex_t>(parents.data(), parents.size()),
         v_first = graph_view.local_vertex_partition_range_first()] __device__(auto edge) {
          parents[std::get<0>(edge) - v_first] = std::get<1>(edge);
        });

      remaining_vertices.resize(thrust::distance(remaining_vertices.begin(), deg0_v_first),
                                handle.get_stream());
    } else {  // all the remaining vertices have a degree greater than 1
      break;
    }
  }

  return parents;
}

}  // namespace cugraph

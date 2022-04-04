/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <tuple>

#include <utilities/high_res_timer.hpp>

namespace cugraph {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_induced_subgraphs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  size_t const* subgraph_offsets /* size == num_subgraphs + 1 */,
  vertex_t const* subgraph_vertices /* size == subgraph_offsets[num_subgraphs] */,
  size_t num_subgraphs,
  bool do_expensive_check)
{
#ifdef TIMING
  HighResTimer hr_timer;
  hr_timer.start("extract_induced_subgraphs");
#endif
  // FIXME: this code is inefficient for the vertices with their local degrees much larger than the
  // number of vertices in the subgraphs (in this case, searching that the subgraph vertices are
  // included in the local neighbors is more efficient than searching the local neighbors are
  // included in the subgraph vertices). We may later add additional code to handle such cases.
  // FIXME: we may consider the performance (speed & memory footprint, hash based approach uses
  // extra-memory) of hash table based and binary search based approaches

  // 1. check input arguments

  if (do_expensive_check) {
    size_t should_be_zero{std::numeric_limits<size_t>::max()};
    size_t num_aggregate_subgraph_vertices{};
    raft::update_host(&should_be_zero, subgraph_offsets, 1, handle.get_stream());
    raft::update_host(
      &num_aggregate_subgraph_vertices, subgraph_offsets + num_subgraphs, 1, handle.get_stream());
    handle.sync_stream();
    CUGRAPH_EXPECTS(should_be_zero == 0,
                    "Invalid input argument: subgraph_offsets[0] should be 0.");

    CUGRAPH_EXPECTS(
      thrust::is_sorted(
        handle.get_thrust_policy(), subgraph_offsets, subgraph_offsets + (num_subgraphs + 1)),
      "Invalid input argument: subgraph_offsets is not sorted.");
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       subgraph_vertices,
                       subgraph_vertices + num_aggregate_subgraph_vertices,
                       [vertex_partition] __device__(auto v) {
                         return !vertex_partition.is_valid_vertex(v) ||
                                !vertex_partition.in_local_vertex_partition_range_nocheck(v);
                       }) == 0,
      "Invalid input argument: subgraph_vertices has invalid vertex IDs.");

    CUGRAPH_EXPECTS(
      thrust::count_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_subgraphs),
        [subgraph_offsets, subgraph_vertices] __device__(auto i) {
          // vertices are sorted and unique
          return !thrust::is_sorted(thrust::seq,
                                    subgraph_vertices + subgraph_offsets[i],
                                    subgraph_vertices + subgraph_offsets[i + 1]) ||
                 (thrust::count_if(
                    thrust::seq,
                    thrust::make_counting_iterator(subgraph_offsets[i]),
                    thrust::make_counting_iterator(subgraph_offsets[i + 1]),
                    [subgraph_vertices, last = subgraph_offsets[i + 1] - 1] __device__(auto i) {
                      return (i != last) && (subgraph_vertices[i] == subgraph_vertices[i + 1]);
                    }) != 0);
        }) == 0,
      "Invalid input argument: subgraph_vertices for each subgraph idx should be sorted in "
      "ascending order and unique.");
  }

  // 2. extract induced subgraphs

  if (multi_gpu) {
    CUGRAPH_FAIL("Unimplemented.");
    return std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                           rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                           rmm::device_uvector<weight_t>(0, handle.get_stream()),
                           rmm::device_uvector<size_t>(0, handle.get_stream()));
  } else {
    // 2-1. Phase 1: calculate memory requirements

    size_t num_aggregate_subgraph_vertices{};
    raft::update_host(
      &num_aggregate_subgraph_vertices, subgraph_offsets + num_subgraphs, 1, handle.get_stream());
    handle.sync_stream();

    rmm::device_uvector<size_t> subgraph_vertex_output_offsets(
      num_aggregate_subgraph_vertices + 1,
      handle.get_stream());  // for each element of subgraph_vertices

    auto edge_partition = edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
      graph_view.local_edge_partition_view());
    // count the numbers of the induced subgraph edges for each vertex in the aggregate subgraph
    // vertex list.
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_aggregate_subgraph_vertices),
      subgraph_vertex_output_offsets.begin(),
      [subgraph_offsets, subgraph_vertices, num_subgraphs, edge_partition] __device__(auto i) {
        auto subgraph_idx = thrust::distance(
          subgraph_offsets + 1,
          thrust::upper_bound(thrust::seq, subgraph_offsets, subgraph_offsets + num_subgraphs, i));
        vertex_t const* indices{nullptr};
        thrust::optional<weight_t const*> weights{thrust::nullopt};
        edge_t local_degree{};
        auto major_offset = edge_partition.major_offset_from_major_nocheck(subgraph_vertices[i]);
        thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_offset);
        // FIXME: this is inefficient for high local degree vertices
        return thrust::count_if(
          thrust::seq,
          indices,
          indices + local_degree,
          [vertex_first = subgraph_vertices + subgraph_offsets[subgraph_idx],
           vertex_last =
             subgraph_vertices + subgraph_offsets[subgraph_idx + 1]] __device__(auto nbr) {
            return thrust::binary_search(thrust::seq, vertex_first, vertex_last, nbr);
          });
      });
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           subgraph_vertex_output_offsets.begin(),
                           subgraph_vertex_output_offsets.end(),
                           subgraph_vertex_output_offsets.begin());

    size_t num_aggregate_edges{};
    raft::update_host(&num_aggregate_edges,
                      subgraph_vertex_output_offsets.data() + num_aggregate_subgraph_vertices,
                      1,
                      handle.get_stream());
    handle.sync_stream();

    // 2-2. Phase 2: find the edges in the induced subgraphs

    rmm::device_uvector<vertex_t> edge_majors(num_aggregate_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> edge_minors(num_aggregate_edges, handle.get_stream());
    auto edge_weights = graph_view.is_weighted()
                          ? std::make_optional<rmm::device_uvector<weight_t>>(num_aggregate_edges,
                                                                              handle.get_stream())
                          : std::nullopt;

    // fill the edge list buffer (to be returned) for each vetex in the aggregate subgraph vertex
    // list (use the offsets computed in the Phase 1)
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_aggregate_subgraph_vertices),
      [subgraph_offsets,
       subgraph_vertices,
       num_subgraphs,
       edge_partition,
       subgraph_vertex_output_offsets = subgraph_vertex_output_offsets.data(),
       edge_majors                    = edge_majors.data(),
       edge_minors                    = edge_minors.data(),
       edge_weights = edge_weights ? thrust::optional<weight_t*>{(*edge_weights).data()}
                                   : thrust::nullopt] __device__(auto i) {
        auto subgraph_idx = thrust::distance(
          subgraph_offsets + 1,
          thrust::upper_bound(
            thrust::seq, subgraph_offsets, subgraph_offsets + num_subgraphs, size_t{i}));
        vertex_t const* indices{nullptr};
        thrust::optional<weight_t const*> weights{thrust::nullopt};
        edge_t local_degree{};
        auto major_offset = edge_partition.major_offset_from_major_nocheck(subgraph_vertices[i]);
        thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_offset);
        if (weights) {
          auto triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_constant_iterator(subgraph_vertices[i]), indices, *weights));
          // FIXME: this is inefficient for high local degree vertices
          thrust::copy_if(
            thrust::seq,
            triplet_first,
            triplet_first + local_degree,
            thrust::make_zip_iterator(thrust::make_tuple(edge_majors, edge_minors, *edge_weights)) +
              subgraph_vertex_output_offsets[i],
            [vertex_first = subgraph_vertices + subgraph_offsets[subgraph_idx],
             vertex_last =
               subgraph_vertices + subgraph_offsets[subgraph_idx + 1]] __device__(auto t) {
              return thrust::binary_search(
                thrust::seq, vertex_first, vertex_last, thrust::get<1>(t));
            });
        } else {
          auto pair_first = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::make_constant_iterator(subgraph_vertices[i]), indices));
          // FIXME: this is inefficient for high local degree vertices
          thrust::copy_if(thrust::seq,
                          pair_first,
                          pair_first + local_degree,
                          thrust::make_zip_iterator(thrust::make_tuple(edge_majors, edge_minors)) +
                            subgraph_vertex_output_offsets[i],
                          [vertex_first = subgraph_vertices + subgraph_offsets[subgraph_idx],
                           vertex_last  = subgraph_vertices +
                                         subgraph_offsets[subgraph_idx + 1]] __device__(auto t) {
                            return thrust::binary_search(
                              thrust::seq, vertex_first, vertex_last, thrust::get<1>(t));
                          });
        }
      });

    rmm::device_uvector<size_t> subgraph_edge_offsets(num_subgraphs + 1, handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   subgraph_offsets,
                   subgraph_offsets + (num_subgraphs + 1),
                   subgraph_vertex_output_offsets.begin(),
                   subgraph_edge_offsets.begin());
#ifdef TIMING
    hr_timer.stop();
    hr_timer.display(std::cout);
#endif
    return std::make_tuple(std::move(edge_majors),
                           std::move(edge_minors),
                           std::move(edge_weights),
                           std::move(subgraph_edge_offsets));
  }
}

}  // namespace cugraph

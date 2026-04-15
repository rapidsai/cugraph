/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

namespace cugraph {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> topological_sort(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  // Topological sort exists only if graph is directed and does not contain any loops
  CUGRAPH_EXPECTS(!graph_view.is_symmetric(), "Invalid input argument: topological sort requires graph to be directed");

  if (do_expensive_check) {
    auto num_self_loops = graph_view.count_self_loops(handle);
    CUGRAPH_EXPECTS(num_self_loops == 0, "Invalid input argument: topological sort requires graph without self loops");

    auto components = strongly_connected_components(handle, graph_view, true);

    thrust::sort(handle.get_thrust_policy(), components.begin(), components.end());
    CUGRAPH_EXPECTS(thrust::unique_count(handle.get_thrust_policy(), components.begin(), components.end()) == components.size(), "Invalid input argument: topological sort requires graph without cycles");

    if constexpr (multi_gpu) {
      std::tie(components, std::ignore) =
        shuffle_ext_vertices(handle, std::move(components), std::vector<arithmetic_device_uvector_t>{});

      thrust::sort(handle.get_thrust_policy(), components.begin(), components.end());
      CUGRAPH_EXPECTS(thrust::unique_count(handle.get_thrust_policy(), components.begin(), components.end()) == components.size(), "Invalid input argument: topological sort requires graph without cycles");
    }
  }

  rmm::device_uvector<vertex_t> frontier_vertices(graph_view.local_vertex_partition_range_size(),
                                                  handle.get_stream());
  auto in_degrees = graph_view.compute_in_degrees(handle);

  frontier_vertices.resize(
    cuda::std::distance(
      frontier_vertices.begin(),
      thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
        thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
        frontier_vertices.begin(),
        cuda::proclaim_return_type<bool>(
          [in_degrees = raft::device_span<edge_t const>(in_degrees.data(), in_degrees.size()),
           v_first    = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
            auto v_offset = v - v_first;
            return in_degrees[v_offset] == 0;
          }))),
    handle.get_stream());

  auto aggregate_frontier_size = 0;

  rmm::device_uvector<vertex_t> topological_levels(graph_view.local_vertex_partition_range_size(),
                                                   handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), topological_levels.begin(), topological_levels.end(), vertex_t{0});

  auto level = 0;
  while (true) {
    aggregate_frontier_size = frontier_vertices.size();
    if constexpr (multi_gpu) {
      aggregate_frontier_size = host_scalar_allreduce(
        handle.get_comms(), aggregate_frontier_size, raft::comms::op_t::SUM, handle.get_stream());
    }
    if (aggregate_frontier_size == 0) { break; }

    key_bucket_view_t<vertex_t, void, multi_gpu, true> frontier(
      handle,
      raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()));

    rmm::device_uvector<vertex_t> dst_vertices(0, handle.get_stream());
    rmm::device_uvector<edge_t> decrement_counts(0, handle.get_stream());
    std::tie(dst_vertices, decrement_counts) =
      cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
        handle,
        graph_view,
        frontier,
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<edge_t>(
          [] __device__(auto src, auto dst, auto, auto, auto) { return edge_t{1}; }),
        reduce_op::plus<edge_t>());

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(dst_vertices.begin(), decrement_counts.begin()),
      thrust::make_zip_iterator(dst_vertices.end(), decrement_counts.end()),
      [in_degrees = raft::device_span<edge_t>(in_degrees.data(), in_degrees.size()),
       v_first    = graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
        auto v_offset        = cuda::std::get<0>(pair) - v_first;
        auto decrement_count = cuda::std::get<1>(pair);
        in_degrees[v_offset] -= decrement_count;
      });

    rmm::device_uvector<vertex_t> new_frontier_vertices(dst_vertices.size(), handle.get_stream());
    
    new_frontier_vertices.resize(
      cuda::std::distance(new_frontier_vertices.begin(),
                          thrust::copy_if(
                            handle.get_thrust_policy(),
                            dst_vertices.begin(),
                            dst_vertices.end(),
                            new_frontier_vertices.begin(),
                            [in_degrees = raft::device_span<edge_t const>(in_degrees.data(), in_degrees.size()),
                            v_first    = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
                              auto v_offset = v - v_first;
                              return in_degrees[v_offset] == 0;
                            })),
      handle.get_stream());
    new_frontier_vertices.shrink_to_fit(handle.get_stream());

    thrust::for_each(handle.get_thrust_policy(),
                     frontier_vertices.begin(),
                     frontier_vertices.end(),
                     [topological_levels = raft::device_span<vertex_t>(topological_levels.data(),
                                                                       topological_levels.size()),
                      v_first            = graph_view.local_vertex_partition_range_first(),
                      level              = level] __device__(auto v) {
                       auto v_offset                = v - v_first;
                       topological_levels[v_offset] = level;
                     });

    frontier_vertices = std::move(new_frontier_vertices);
    level++;
  }

  return topological_levels;
}

}  // namespace cugraph

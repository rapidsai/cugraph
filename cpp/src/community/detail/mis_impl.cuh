
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <community/detail/mis.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  vertex_t local_vtx_partitoin_size = graph_view.local_vertex_partition_range_size();

  rmm::device_uvector<vertex_t> remaining_vertices(local_vtx_partitoin_size, handle.get_stream());

  auto vertex_begin =
    thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first());
  auto vertex_end = thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last());

  // Compute out-degree
  auto out_degrees = graph_view.compute_out_degrees(handle);

  // Vertices with non-zero out-degree are possible candidates for MIS.
  remaining_vertices.resize(
    thrust::distance(remaining_vertices.begin(),
                     thrust::copy_if(handle.get_thrust_policy(),
                                     vertex_begin,
                                     vertex_end,
                                     out_degrees.begin(),
                                     remaining_vertices.begin(),
                                     [] __device__(auto deg) { return deg > 0; })),
    handle.get_stream());

  // Set ID of each vertex as its rank
  rmm::device_uvector<vertex_t> ranks(local_vtx_partitoin_size, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), vertex_begin, vertex_end, ranks.begin());

  // Set ranks of zero out-degree vetices to std::numeric_limits<vertex_t>::lowest()
  thrust::for_each(
    handle.get_thrust_policy(),
    vertex_begin,
    vertex_end,
    [out_degrees = raft::device_span<edge_t const>(out_degrees.data(), out_degrees.size()),
     ranks       = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
     v_first     = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
      auto v_offset = v - v_first;
      if (out_degrees[v_offset] == 0) { ranks[v_offset] = std::numeric_limits<vertex_t>::lowest(); }
    });

  out_degrees.resize(0, handle.get_stream());
  out_degrees.shrink_to_fit(handle.get_stream());

  thrust::default_random_engine g;
  size_t seed = 0;
  if constexpr (multi_gpu) { seed = handle.get_comms().get_rank(); }
  g.seed(seed);

  size_t loop_counter = 0;
  while (true) {
    loop_counter++;

    // Copy ranks into temporary vector to begin with

    rmm::device_uvector<vertex_t> temporary_ranks(local_vtx_partitoin_size, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), ranks.begin(), ranks.end(), temporary_ranks.begin());

    // Select a random set of candidate vertices
    // FIXME: use common utility function to select a subset of remaining vertices
    // and for MG extension, select from disributed array remaining vertices
    thrust::shuffle(
      handle.get_thrust_policy(), remaining_vertices.begin(), remaining_vertices.end(), g);

    vertex_t nr_candidates =
      (remaining_vertices.size() < 1024)
        ? remaining_vertices.size()
        : std::min(static_cast<vertex_t>((0.50 + 0.25 * loop_counter) * remaining_vertices.size()),
                   static_cast<vertex_t>(remaining_vertices.size()));

    // Set temporary ranks of non-candidate vertices to std::numeric_limits<vertex_t>::lowest()
    thrust::for_each(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end() - nr_candidates,
      [temporary_ranks =
         raft::device_span<vertex_t>(temporary_ranks.data(), temporary_ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        //
        // if rank of a non-candidate vertex is not std::numeric_limits<vertex_t>::max() (i.e. the
        // vertex is not already in MIS), set it to std::numeric_limits<vertex_t>::lowest()
        //
        auto v_offset = v - v_first;
        if (temporary_ranks[v_offset] < std::numeric_limits<vertex_t>::max()) {
          temporary_ranks[v_offset] = std::numeric_limits<vertex_t>::lowest();
        }
      });

    // Caches for ranks
    edge_src_property_t<GraphViewType, vertex_t> src_rank_cache(handle);
    edge_dst_property_t<GraphViewType, vertex_t> dst_rank_cache(handle);

    // Update rank caches with temporary ranks
    if constexpr (multi_gpu) {
      src_rank_cache = edge_src_property_t<GraphViewType, vertex_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, vertex_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, temporary_ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, temporary_ranks.begin(), dst_rank_cache);
    }

    //
    // Find maximum rank outgoing neighbor for each vertex
    // (In case of Leiden decision graph, each vertex has at most one outgoing edge)
    //

    rmm::device_uvector<vertex_t> max_outgoing_ranks(local_vtx_partitoin_size, handle.get_stream());

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      multi_gpu
        ? src_rank_cache.view()
        : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(temporary_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                    temporary_ranks.data(), vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return dst_rank; },
      std::numeric_limits<vertex_t>::lowest(),
      cugraph::reduce_op::maximum<vertex_t>{},
      max_outgoing_ranks.begin());

    //
    // Find maximum rank incoming neighbor for each vertex
    //

    rmm::device_uvector<vertex_t> max_incoming_ranks(local_vtx_partitoin_size, handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      multi_gpu
        ? src_rank_cache.view()
        : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(temporary_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                    temporary_ranks.data(), vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return src_rank; },
      std::numeric_limits<vertex_t>::lowest(),
      cugraph::reduce_op::maximum<vertex_t>{},
      max_incoming_ranks.begin());

    temporary_ranks.resize(0, handle.get_stream());
    temporary_ranks.shrink_to_fit(handle.get_stream());

    //
    // Compute max of outgoing and incoming neighbors
    //
    thrust::transform(handle.get_thrust_policy(),
                      max_incoming_ranks.begin(),
                      max_incoming_ranks.end(),
                      max_outgoing_ranks.begin(),
                      max_outgoing_ranks.begin(),
                      thrust::maximum<vertex_t>());

    max_incoming_ranks.resize(0, handle.get_stream());
    max_incoming_ranks.shrink_to_fit(handle.get_stream());

    //
    // If the max neighbor of a vertex is already in MIS (i.e. has rank
    // std::numeric_limits<vertex_t>::max()), discard it, otherwise,
    // include the vertex if it has larger rank than its maximum rank neighbor
    //
    auto last = thrust::remove_if(
      handle.get_thrust_policy(),
      remaining_vertices.end() - nr_candidates,
      remaining_vertices.end(),
      [max_rank_neighbor_first = max_outgoing_ranks.begin(),
       ranks                   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset          = v - v_first;
        auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);
        auto rank_of_v         = ranks[v_offset];

        if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
          // Maximum rank neighbor is alreay in MIS
          // Discard current vertex by setting its rank to
          // std::numeric_limits<vertex_t>::lowest()
          ranks[v_offset] = std::numeric_limits<vertex_t>::lowest();
          return true;
        }

        if (rank_of_v >= max_neighbor_rank) {
          // Include v and set its rank to std::numeric_limits<vertex_t>::max()
          ranks[v_offset] = std::numeric_limits<vertex_t>::max();
          return true;
        }
        return false;
      });

    max_outgoing_ranks.resize(0, handle.get_stream());
    max_outgoing_ranks.shrink_to_fit(handle.get_stream());

    remaining_vertices.resize(thrust::distance(remaining_vertices.begin(), last),
                              handle.get_stream());
    remaining_vertices.shrink_to_fit(handle.get_stream());

    vertex_t nr_remaining_vertices_to_check = remaining_vertices.size();
    if (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    if (nr_remaining_vertices_to_check == 0) { break; }
  }

  // Count number of vertices included in MIS

  vertex_t nr_vertices_included_in_mis = thrust::count_if(
    handle.get_thrust_policy(), ranks.begin(), ranks.end(), [] __device__(auto v_rank) {
      return v_rank >= std::numeric_limits<vertex_t>::max();
    });

  // Build MIS and return
  rmm::device_uvector<vertex_t> mis(nr_vertices_included_in_mis, handle.get_stream());
  thrust::copy_if(
    handle.get_thrust_policy(),
    vertex_begin,
    vertex_end,
    ranks.begin(),
    mis.begin(),
    [] __device__(auto v_rank) { return v_rank >= std::numeric_limits<vertex_t>::max(); });

  ranks.resize(0, handle.get_stream());
  ranks.shrink_to_fit(handle.get_stream());
  return mis;
}
}  // namespace detail
}  // namespace cugraph

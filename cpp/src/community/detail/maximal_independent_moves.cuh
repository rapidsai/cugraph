
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "maximal_independent_moves.hpp"

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/prims/extract_transform_if_v.cuh>
#include <cugraph/prims/fill_edge_src_dst_property.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/transform_if_v.cuh>
#include <cugraph/prims/transform_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_property.hpp>
#include <cugraph/vertex_property_device_view.cuh>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/for_each.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <optional>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> maximal_independent_moves(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  vertex_t local_vtx_partitoin_size = graph_view.local_vertex_partition_range_size();

  // Compute out-degree
  auto out_degrees = compute_out_degrees(handle, graph_view);

  // Only vertices with non-zero out-degree are possible can move
  auto remaining_vertices = extract_transform_if_v(
    handle,
    graph_view,
    out_degrees.view(),
    cuda::proclaim_return_type<vertex_t>([] __device__(auto v, auto) { return v; }),
    cuda::proclaim_return_type<bool>([] __device__(auto, auto deg) { return deg > 0; }));

  // Set ID of each vertex as its rank
  vertex_property_t<vertex_t, vertex_t> ranks(handle, graph_view);
  transform_v(handle, graph_view, cuda::std::identity{}, ranks.mutable_view());

  // Set ranks of zero out-degree vetices to std::numeric_limits<vertex_t>::lowest()
  transform_if_v(handle,
                 graph_view,
                 out_degrees.view(),
                 cuda::proclaim_return_type<vertex_t>(
                   [] __device__(auto) { return std::numeric_limits<vertex_t>::lowest(); }),
                 cuda::proclaim_return_type<bool>([] __device__(auto deg) { return deg == 0; }),
                 ranks.mutable_view());

  out_degrees = vertex_property_t<vertex_t, edge_t>(handle);

  size_t loop_counter = 0;
  while (true) {
    loop_counter++;

    // Copy ranks into temporary vector to begin with

    vertex_property_t<vertex_t, vertex_t> temporary_ranks(handle, graph_view);
    copy_v(handle, graph_view, ranks.view(), temporary_ranks.mutable_view());

    // Select a random set of candidate vertices

    vertex_t nr_remaining_vertices_to_check = remaining_vertices.size();
    if constexpr (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    vertex_t nr_candidates = (nr_remaining_vertices_to_check < 1024)
                               ? nr_remaining_vertices_to_check
                               : std::min(static_cast<vertex_t>((0.50 + 0.25 * loop_counter) *
                                                                nr_remaining_vertices_to_check),
                                          nr_remaining_vertices_to_check);

    // FIXME: Can we improve performance here?
    // FIXME: if(nr_remaining_vertices_to_check < 1024), may avoid calling select_random_vertices
    auto d_sampled_vertices =
      cugraph::select_random_vertices(handle,
                                      rng_state,
                                      graph_view,
                                      std::make_optional(raft::device_span<vertex_t const>{
                                        remaining_vertices.data(), remaining_vertices.size()}),
                                      nr_candidates,
                                      false,
                                      true);

    rmm::device_uvector<vertex_t> non_candidate_vertices(
      remaining_vertices.size() - d_sampled_vertices.size(), handle.get_stream());

    thrust::set_difference(handle.get_thrust_policy(),
                           remaining_vertices.begin(),
                           remaining_vertices.end(),
                           d_sampled_vertices.begin(),
                           d_sampled_vertices.end(),
                           non_candidate_vertices.begin());

    // Set temporary ranks of non-candidate vertices to std::numeric_limits<vertex_t>::lowest()
    thrust::for_each(
      handle.get_thrust_policy(),
      non_candidate_vertices.begin(),
      non_candidate_vertices.end(),
      [temporary_ranks = detail::vertex_property_device_view_t{temporary_ranks.mutable_view()},
       v_first         = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        //
        // if rank of a non-candidate vertex is not std::numeric_limits<vertex_t>::max() (i.e. the
        // vertex is not already in MIS), set it to std::numeric_limits<vertex_t>::lowest()
        //
        auto v_offset = v - v_first;
        if (temporary_ranks.get(v_offset) < std::numeric_limits<vertex_t>::max()) {
          temporary_ranks.set(v_offset, std::numeric_limits<vertex_t>::lowest());
        }
      });

    // Caches for ranks
    edge_src_property_t<vertex_t, vertex_t> src_rank_cache(handle);
    edge_dst_property_t<vertex_t, vertex_t> dst_rank_cache(handle);

    // Update rank caches with temporary ranks
    if constexpr (multi_gpu) {
      src_rank_cache = edge_src_property_t<vertex_t, vertex_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<vertex_t, vertex_t>(handle, graph_view);
      update_edge_src_property(
        handle, graph_view, temporary_ranks.view(), src_rank_cache.mutable_view());
      update_edge_dst_property(
        handle, graph_view, temporary_ranks.view(), dst_rank_cache.mutable_view());
    }

    //
    // Find maximum rank outgoing neighbor for each vertex
    //

    rmm::device_uvector<vertex_t> max_outgoing_ranks(local_vtx_partitoin_size, handle.get_stream());

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      multi_gpu ? src_rank_cache.view()
                : make_edge_src_property_view(graph_view, temporary_ranks.view()),
      multi_gpu ? dst_rank_cache.view()
                : make_edge_dst_property_view(graph_view, temporary_ranks.view()),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return dst_rank; },
      std::numeric_limits<vertex_t>::lowest(),
      cugraph::reduce_op::maximum<vertex_t>{},
      vertex_property_view_t{max_outgoing_ranks.begin(),
                             graph_view.local_vertex_partition_range_size()});

    //
    // Find maximum rank incoming neighbor for each vertex
    //

    rmm::device_uvector<vertex_t> max_incoming_ranks(local_vtx_partitoin_size, handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      multi_gpu ? src_rank_cache.view()
                : make_edge_src_property_view(graph_view, temporary_ranks.view()),
      multi_gpu ? dst_rank_cache.view()
                : make_edge_dst_property_view(graph_view, temporary_ranks.view()),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return src_rank; },
      std::numeric_limits<vertex_t>::lowest(),
      cugraph::reduce_op::maximum<vertex_t>{},
      vertex_property_view_t{max_incoming_ranks.begin(),
                             graph_view.local_vertex_partition_range_size()});

    temporary_ranks = vertex_property_t<vertex_t, vertex_t>(handle);

    //
    // Compute max of outgoing and incoming neighbors
    //
    thrust::transform(handle.get_thrust_policy(),
                      max_incoming_ranks.begin(),
                      max_incoming_ranks.end(),
                      max_outgoing_ranks.begin(),
                      max_outgoing_ranks.begin(),
                      cuda::maximum<vertex_t>());

    max_incoming_ranks.resize(0, handle.get_stream());
    max_incoming_ranks.shrink_to_fit(handle.get_stream());

    //
    // If the max neighbor of a vertex is already in MIS (i.e. has rank
    // std::numeric_limits<vertex_t>::max()), discard it, otherwise,
    // include the vertex if it has larger rank than its maximum rank neighbor
    //
    auto last = thrust::remove_if(
      handle.get_thrust_policy(),
      d_sampled_vertices.begin(),
      d_sampled_vertices.end(),
      [max_rank_neighbor_first = max_outgoing_ranks.begin(),
       ranks                   = detail::vertex_property_device_view_t{ranks.mutable_view()},
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset          = v - v_first;
        auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);
        auto rank_of_v         = ranks.get(v_offset);

        if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
          // Maximum rank neighbor is alreay in MIS
          // Discard current vertex by setting its rank to
          // std::numeric_limits<vertex_t>::lowest()
          ranks.set(v_offset, std::numeric_limits<vertex_t>::lowest());
          return true;
        }

        if (rank_of_v >= max_neighbor_rank) {
          // Include v and set its rank to std::numeric_limits<vertex_t>::max()
          ranks.set(v_offset, std::numeric_limits<vertex_t>::max());
          return true;
        }
        return false;
      });

    max_outgoing_ranks.resize(0, handle.get_stream());
    max_outgoing_ranks.shrink_to_fit(handle.get_stream());

    d_sampled_vertices.resize(cuda::std::distance(d_sampled_vertices.begin(), last),
                              handle.get_stream());
    d_sampled_vertices.shrink_to_fit(handle.get_stream());

    remaining_vertices.resize(non_candidate_vertices.size() + d_sampled_vertices.size(),
                              handle.get_stream());
    remaining_vertices.shrink_to_fit(handle.get_stream());

    // merge non-candidate and remaining candidate vertices
    thrust::merge(handle.get_thrust_policy(),
                  non_candidate_vertices.begin(),
                  non_candidate_vertices.end(),
                  d_sampled_vertices.begin(),
                  d_sampled_vertices.end(),
                  remaining_vertices.begin());

    nr_remaining_vertices_to_check = remaining_vertices.size();
    if constexpr (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    if (nr_remaining_vertices_to_check == 0) { break; }
  }

  // Build MIS and return
  return extract_transform_if_v(
    handle,
    graph_view,
    ranks.view(),
    cuda::proclaim_return_type<vertex_t>([] __device__(auto v, auto) { return v; }),
    cuda::proclaim_return_type<bool>(
      [] __device__(auto, auto v_rank) { return v_rank >= std::numeric_limits<vertex_t>::max(); }));
}
}  // namespace detail

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/shuffle_wrappers.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/per_v_transform_reduce_if_incoming_outgoing_e.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> maximal_independent_set(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  // A vertex is adjacent to both its incoming and its outgoing neighbors. For a symmetric graph
  // the outgoing edges alone cover both, otherwise the incoming edges are traversed as well.
  auto symmetric = graph_view.is_symmetric();

  vertex_t local_vtx_partition_size = graph_view.local_vertex_partition_range_size();
  auto v_first                      = graph_view.local_vertex_partition_range_first();

  auto vertex_begin = thrust::make_counting_iterator(v_first);
  auto vertex_end = thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last());

  auto segment_offsets = graph_view.local_vertex_partition_segment_offsets();

  //
  // Degree segment offsets of the local vertex partition, and the vertices in the degree
  // ordering the offsets refer to. Renumbered vertices are already sorted by decreasing out
  // degree, so for a symmetric graph the segment offsets of the graph apply to the local vertex
  // partition range as is. Otherwise, the local vertices are sorted by degree here and the
  // segment offsets are computed from the sorted degrees, using the same thresholds renumbering
  // would have used.
  //
  std::vector<vertex_t> h_segment_offsets{};
  std::optional<rmm::device_uvector<vertex_t>> sorted_vertices{std::nullopt};

  if (segment_offsets && symmetric) {
    h_segment_offsets = *segment_offsets;
  } else {
    auto degrees = graph_view.compute_out_degrees(handle);

    if (!symmetric) {
      // The degree of a vertex is the number of its incoming and outgoing neighbors
      auto in_degrees = graph_view.compute_in_degrees(handle);
      thrust::transform(handle.get_thrust_policy(),
                        degrees.begin(),
                        degrees.end(),
                        in_degrees.begin(),
                        degrees.begin(),
                        cuda::std::plus<edge_t>{});
    }

    sorted_vertices = rmm::device_uvector<vertex_t>(local_vtx_partition_size, handle.get_stream());
    cugraph::sequence(
      handle.get_thrust_policy(), (*sorted_vertices).begin(), (*sorted_vertices).end(), v_first);

    // sort local vertices by degree (descending)
    thrust::sort_by_key(handle.get_thrust_policy(),
                        degrees.begin(),
                        degrees.end(),
                        (*sorted_vertices).begin(),
                        cuda::std::greater<edge_t>());

    static_assert(num_sparse_segments_per_vertex_partition == 3);

    h_segment_offsets.resize(num_sparse_segments_per_vertex_partition + 2);
    h_segment_offsets.front() = vertex_t{0};
    h_segment_offsets.back()  = local_vtx_partition_size;
    for (size_t i = 0; i < num_sparse_segments_per_vertex_partition; ++i) {
      auto threshold = static_cast<edge_t>((i == 0)   ? mid_degree_threshold  // high, mid
                                           : (i == 1) ? low_degree_threshold  // mid, low
                                                      : size_t{1});           // low, zero
      h_segment_offsets[i + 1] =
        static_cast<vertex_t>(thrust::count_if(handle.get_thrust_policy(),
                                               degrees.begin(),
                                               degrees.end(),
                                               is_greater_than_or_equal_to_t<edge_t>{threshold}));
    }

    degrees.resize(0, handle.get_stream());
    degrees.shrink_to_fit(handle.get_stream());
  }

  // Vertices with degree zero are in the last segment
  vertex_t isolated_v_start = *(h_segment_offsets.rbegin() + 1);

  //
  // Rank of a vertex is its priority. Ranks are unique, so two adjacent vertices can never be
  // included in the MIS in the same iteration. They are assigned in the degree ordering the
  // segment offsets refer to, and mapped back to the local vertex partition order below.
  //
  rmm::device_uvector<vertex_t> sorted_vertex_ranks(local_vtx_partition_size, handle.get_stream());

  if constexpr (multi_gpu) {
    //
    // Set ID of each vertex as its rank. Vertices are assigned to GPUs by hashing their external
    // IDs, and each GPU owns one contiguous block of the renumbered ID space while sorting its
    // own vertices by degree. A rank is therefore decided first by the (random) GPU block and
    // only then by degree, which already amounts to a random permutation coarsened to comm_size
    // levels. Perturbing the ranks any further only adds work.
    //
    if (sorted_vertices) {
      thrust::copy(handle.get_thrust_policy(),
                   (*sorted_vertices).begin(),
                   (*sorted_vertices).end(),
                   sorted_vertex_ranks.begin());
    } else {
      cugraph::sequence(handle.get_thrust_policy(),
                        sorted_vertex_ranks.begin(),
                        sorted_vertex_ranks.end(),
                        v_first);
    }
  } else {
    //
    // Set a random permutation of each degree segment as the ranks of the segment. This keeps the
    // degree bands intact, so that the low degree vertices still get high priority (which yields
    // a larger MIS), while randomizing the order within a band, which breaks the long dependency
    // chains a monotone ordering creates in high diameter graphs.
    //
    for (size_t i = 0; i + 1 < h_segment_offsets.size(); ++i) {
      auto segment_first = h_segment_offsets[i];
      auto segment_last  = std::min(h_segment_offsets[i + 1], isolated_v_start);
      if (segment_last <= segment_first) { continue; }  // empty segment, or the zero degree one

      auto permuted_ranks = permute_range<vertex_t>(
        handle, rng_state, v_first + segment_first, segment_last - segment_first);

      thrust::copy(handle.get_thrust_policy(),
                   permuted_ranks.begin(),
                   permuted_ranks.end(),
                   sorted_vertex_ranks.begin() + segment_first);
    }
  }

  // Vertices with degree zero are always part of MIS, and they are the last degree segment
  thrust::fill(handle.get_thrust_policy(),
               sorted_vertex_ranks.begin() + isolated_v_start,
               sorted_vertex_ranks.end(),
               std::numeric_limits<vertex_t>::max());

  // Map the ranks from the degree ordering back to the local vertex partition order. The degree
  // ordering of the renumbered vertices is the local vertex partition order.
  rmm::device_uvector<vertex_t> ranks(0, handle.get_stream());

  if (sorted_vertices) {
    ranks.resize(local_vtx_partition_size, handle.get_stream());

    thrust::scatter(
      handle.get_thrust_policy(),
      sorted_vertex_ranks.begin(),
      sorted_vertex_ranks.end(),
      thrust::make_transform_iterator((*sorted_vertices).begin(), shift_left_t<vertex_t>{v_first}),
      ranks.begin());

    sorted_vertex_ranks.resize(0, handle.get_stream());
    sorted_vertex_ranks.shrink_to_fit(handle.get_stream());
    sorted_vertices.reset();
  } else {
    ranks = std::move(sorted_vertex_ranks);
  }

  //
  // Vertices with degree zero are already part of MIS, the rest are to be checked
  //
  rmm::device_uvector<vertex_t> remaining_vertices(isolated_v_start, handle.get_stream());

  remaining_vertices.resize(
    cuda::std::distance(
      remaining_vertices.begin(),
      thrust::copy_if(handle.get_thrust_policy(),
                      vertex_begin,
                      vertex_end,
                      ranks.begin(),
                      remaining_vertices.begin(),
                      is_less_than_to_t<vertex_t>{std::numeric_limits<vertex_t>::max()})),
    handle.get_stream());

  // Only the ranks of the undecided vertices are queried, and they are kept in the front of
  // remaining_vertices.
  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle, 1);

  // Caches for ranks. The source cache is only needed to traverse the incoming edges of an
  // asymmetric graph.
  edge_src_property_t<vertex_t, vertex_t> src_rank_cache(handle);
  edge_dst_property_t<vertex_t, vertex_t> dst_rank_cache(handle);

  size_t loop_counter                           = 0;
  vertex_t nr_remaining_local_vertices_to_check = remaining_vertices.size();

  while (true) {
    loop_counter++;

    auto num_processed_vertices = remaining_vertices.size() - nr_remaining_local_vertices_to_check;

    if constexpr (multi_gpu) {
      if (loop_counter == 1) {
        // Update the rank of every edge endpoint during the first iteration
        dst_rank_cache = edge_dst_property_t<vertex_t, vertex_t>(handle, graph_view);
        update_edge_dst_property(handle, graph_view, ranks.begin(), dst_rank_cache.mutable_view());

        if (!symmetric) {
          src_rank_cache = edge_src_property_t<vertex_t, vertex_t>(handle, graph_view);
          update_edge_src_property(
            handle, graph_view, ranks.begin(), src_rank_cache.mutable_view());
        }
      } else {
        // Update the ranks of the vertices decided in the previous iteration only. They are the
        // tail of remaining_vertices, and thrust::stable_partition kept them sorted.
        auto processed_vertex_first =
          remaining_vertices.begin() + nr_remaining_local_vertices_to_check;

        rmm::device_uvector<vertex_t> processed_ranks(num_processed_vertices, handle.get_stream());
        thrust::transform(handle.get_thrust_policy(),
                          processed_vertex_first,
                          processed_vertex_first + num_processed_vertices,
                          processed_ranks.begin(),
                          cuda::proclaim_return_type<vertex_t>(
                            [ranks = raft::device_span<vertex_t const>(ranks.data(), ranks.size()),
                             v_first = v_first] __device__(auto v) { return ranks[v - v_first]; }));

        // FIXME: Since the ranks being updated are either std::numeric_limits<vertex_t>::max() or
        // std::numeric_limits<vertex_t>::lowest(), explore 'fill_edge_dst_property' which is
        // faster
        update_edge_dst_property(handle,
                                 graph_view,
                                 processed_vertex_first,
                                 processed_vertex_first + num_processed_vertices,
                                 processed_ranks.begin(),
                                 dst_rank_cache.mutable_view());

        if (!symmetric) {
          update_edge_src_property(handle,
                                   graph_view,
                                   processed_vertex_first,
                                   processed_vertex_first + num_processed_vertices,
                                   processed_ranks.begin(),
                                   src_rank_cache.mutable_view());
        }
      }
    }

    remaining_vertices.resize(nr_remaining_local_vertices_to_check, handle.get_stream());
    remaining_vertices.shrink_to_fit(handle.get_stream());

    vertex_frontier.bucket(0).clear();
    vertex_frontier.bucket(0).insert(remaining_vertices.begin(), remaining_vertices.end());

    //
    // Find maximum rank outgoing neighbor for each undecided vertex
    //

    rmm::device_uvector<vertex_t> max_neighbor_ranks(remaining_vertices.size(),
                                                     handle.get_stream());

    if ((loop_counter == 1) && !multi_gpu && symmetric) {
      // Every vertex is undecided in the first iteration, so instead of the maximum rank
      // neighbor, stop the traversal of a vertex as soon as a higher ranked neighbor is found.
      per_v_transform_reduce_if_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        make_edge_src_property_view<vertex_t, vertex_t>(graph_view, ranks.begin(), ranks.size()),
        make_edge_dst_property_view<vertex_t, vertex_t>(graph_view, ranks.begin(), ranks.size()),
        edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
          return dst_rank;
        },
        std::numeric_limits<vertex_t>::lowest(),
        cugraph::reduce_op::any<vertex_t>{},
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
          return src_rank < dst_rank;
        },
        max_neighbor_ranks.begin());
    } else {
      per_v_transform_reduce_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        multi_gpu ? dst_rank_cache.view()
                  : make_edge_dst_property_view<vertex_t, vertex_t>(
                      graph_view, ranks.begin(), ranks.size()),
        edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
          return dst_rank;
        },
        std::numeric_limits<vertex_t>::lowest(),
        cugraph::reduce_op::maximum<vertex_t>{},
        max_neighbor_ranks.begin());
    }

    if (!symmetric) {
      //
      // Find maximum rank incoming neighbor for each vertex, and reduce the two directions into
      // the maximum rank neighbor of each undecided vertex. This pass cannot be restricted to the
      // frontier, as per_v_transform_reduce_incoming_e only takes a key list with transposed
      // storage.
      //
      rmm::device_uvector<vertex_t> max_incoming_ranks(local_vtx_partition_size,
                                                       handle.get_stream());

      per_v_transform_reduce_incoming_e(
        handle,
        graph_view,
        multi_gpu ? src_rank_cache.view()
                  : make_edge_src_property_view<vertex_t, vertex_t>(
                      graph_view, ranks.begin(), ranks.size()),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
          return src_rank;
        },
        std::numeric_limits<vertex_t>::lowest(),
        cugraph::reduce_op::maximum<vertex_t>{},
        max_incoming_ranks.begin());

      thrust::transform(handle.get_thrust_policy(),
                        remaining_vertices.begin(),
                        remaining_vertices.end(),
                        max_neighbor_ranks.begin(),
                        max_neighbor_ranks.begin(),
                        cuda::proclaim_return_type<vertex_t>(
                          [max_incoming_ranks = raft::device_span<vertex_t const>(
                             max_incoming_ranks.data(), max_incoming_ranks.size()),
                           v_first = v_first] __device__(auto v, auto max_outgoing_rank) {
                            auto max_incoming_rank = max_incoming_ranks[v - v_first];
                            return (max_outgoing_rank > max_incoming_rank) ? max_outgoing_rank
                                                                           : max_incoming_rank;
                          }));
    }

    //
    // If the max neighbor of a vertex is already in MIS (i.e. has rank
    // std::numeric_limits<vertex_t>::max()), discard it, otherwise,
    // include the vertex if it has larger rank than its maximum rank neighbor. The vertices that
    // are still undecided are kept in the front of remaining_vertices.
    //
    auto max_rank_vertex_pair_first =
      thrust::make_zip_iterator(max_neighbor_ranks.begin(), remaining_vertices.begin());

    auto last = thrust::stable_partition(
      handle.get_thrust_policy(),
      max_rank_vertex_pair_first,
      max_rank_vertex_pair_first + remaining_vertices.size(),
      [ranks   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
       v_first = v_first] __device__(auto max_neighbor_rank_and_v) {
        auto max_neighbor_rank = cuda::std::get<0>(max_neighbor_rank_and_v);
        auto v_offset          = cuda::std::get<1>(max_neighbor_rank_and_v) - v_first;
        auto rank_of_v         = ranks[v_offset];

        if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
          // Maximum rank neighbor is alreay in MIS
          // Discard current vertex by setting its rank to
          // std::numeric_limits<vertex_t>::lowest()
          ranks[v_offset] = std::numeric_limits<vertex_t>::lowest();
          return false;
        }

        if (rank_of_v >= max_neighbor_rank) {
          // Include v and set its rank to std::numeric_limits<vertex_t>::max()
          ranks[v_offset] = std::numeric_limits<vertex_t>::max();
          return false;
        }
        return true;
      });

    nr_remaining_local_vertices_to_check =
      static_cast<vertex_t>(cuda::std::distance(max_rank_vertex_pair_first, last));

    max_neighbor_ranks.resize(0, handle.get_stream());
    max_neighbor_ranks.shrink_to_fit(handle.get_stream());

    vertex_t nr_remaining_vertices_to_check = nr_remaining_local_vertices_to_check;
    if constexpr (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    if (nr_remaining_vertices_to_check == 0) { break; }
  }

  // Count number of vertices included in MIS

  vertex_t nr_vertices_included_in_mis =
    thrust::count_if(handle.get_thrust_policy(),
                     ranks.begin(),
                     ranks.end(),
                     is_greater_than_or_equal_to_t<vertex_t>{std::numeric_limits<vertex_t>::max()});

  // Build MIS and return
  rmm::device_uvector<vertex_t> mis(nr_vertices_included_in_mis, handle.get_stream());
  thrust::copy_if(handle.get_thrust_policy(),
                  vertex_begin,
                  vertex_end,
                  ranks.begin(),
                  mis.begin(),
                  is_greater_than_or_equal_to_t<vertex_t>{std::numeric_limits<vertex_t>::max()});

  ranks.resize(0, handle.get_stream());
  ranks.shrink_to_fit(handle.get_stream());
  return mis;
}
}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> maximal_independent_set(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  return detail::maximal_independent_set(handle, graph_view, rng_state);
}

}  // namespace cugraph

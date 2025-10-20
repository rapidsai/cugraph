
/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh" // FIXME: remove if unused
#include "prims/per_v_transform_reduce_if_incoming_outgoing_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
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
rmm::device_uvector<vertex_t> maximal_independent_set(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{

  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  vertex_t local_vtx_partition_size = graph_view.local_vertex_partition_range_size();
  
  auto vertex_begin =
    thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first());

  auto vertex_end = thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last());

  // Set ID of each vertex as its rank
  rmm::device_uvector<vertex_t> ranks(local_vtx_partition_size, handle.get_stream());

  auto segment_offsets = graph_view.local_vertex_partition_segment_offsets();

  vertex_t isolated_v_start = multi_gpu ? segment_offsets->data()[4] : segment_offsets->data()[3];
  // Only the non zero degree vertices are part of the initial 'remaining_vertices' list
  rmm::device_uvector<vertex_t> remaining_vertices(isolated_v_start, handle.get_stream());

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(graph_view.local_vertex_partition_range_size()),
    [
      isolated_v_start = multi_gpu ? segment_offsets->data()[4] : segment_offsets->data()[3],
      ranks = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
      remaining_vertices = raft::device_span<vertex_t>(remaining_vertices.data(), remaining_vertices.size()),
      v_first = graph_view.local_vertex_partition_range_first()] __device__(auto idx) {
      
      
      if (idx < isolated_v_start) {
        // initializing the ranks array
        ranks[idx] = v_first + idx;
        // initializing the remaining vertices array
        remaining_vertices[idx] = v_first + idx;
      } else {
        // zero-degree vertices are always part of the MIS
        ranks[idx] = std::numeric_limits<vertex_t>::max();
      }

  });

  auto num_buckets = 1;

  vertex_frontier_t<vertex_t, void, GraphViewType::is_multi_gpu, true> vertex_frontier(handle,
                                                                                       num_buckets);
  
  size_t loop_counter = 0;
  vertex_t nr_remaining_vertices_to_check = remaining_vertices.size();
  vertex_t nr_remaining_local_vertices_to_check = remaining_vertices.size();
  edge_dst_property_t<vertex_t, vertex_t> dst_rank_cache(handle);


  while (true) {
    loop_counter++;

    auto num_processed_vertices = remaining_vertices.size() - nr_remaining_local_vertices_to_check;

    if constexpr (multi_gpu) {
      if (loop_counter == 1) {
        // Update the property of all edge endpoints during the
        // first iteration
        dst_rank_cache = edge_dst_property_t<vertex_t, vertex_t>(handle, graph_view);

        update_edge_dst_property(
          handle, graph_view, ranks.begin(), dst_rank_cache.mutable_view());

      } else {
        // Only update the property of endpoints that had their ranks modified       
        rmm::device_uvector<vertex_t> processed_ranks(
          num_processed_vertices, handle.get_stream());

        auto pair_idx_processed_vertex_first = thrust::make_zip_iterator(
          thrust::make_counting_iterator<size_t>(0),
          remaining_vertices.begin() + nr_remaining_local_vertices_to_check
        );
        
        thrust::for_each(
          handle.get_thrust_policy(),
          pair_idx_processed_vertex_first,
          pair_idx_processed_vertex_first + num_processed_vertices,
          [processed_ranks =
            raft::device_span<vertex_t>(processed_ranks.data(), processed_ranks.size()),
          ranks =
            raft::device_span<vertex_t>(ranks.data(), ranks.size()),
          v_first = graph_view.local_vertex_partition_range_first()] __device__(auto pair_idx_v) {
            
            auto idx = thrust::get<0>(pair_idx_v);
            auto v = thrust::get<1>(pair_idx_v);
            auto v_offset          = v - v_first;

            processed_ranks[idx] = ranks[v_offset];
        });

        // Only update a subset of the graph edge dst property values

        // FIXME: Since we know that the property being updated are either
        // std::numeric_limits<vertex_t>::max() or std::numeric_limits<vertex_t>::min(),
        // explore 'fill_edge_dst_property' which is faster
        update_edge_dst_property(
          handle,
          graph_view,
          remaining_vertices.begin() + nr_remaining_local_vertices_to_check,
          remaining_vertices.end(),
          processed_ranks.begin(),
          dst_rank_cache.mutable_view()
        );

      }
    }

    rmm::device_uvector<vertex_t> max_outgoing_ranks(
      nr_remaining_local_vertices_to_check, handle.get_stream());

    remaining_vertices.resize(nr_remaining_local_vertices_to_check,
                              handle.get_stream());
    remaining_vertices.shrink_to_fit(handle.get_stream());

    vertex_frontier.bucket(0).clear();

    vertex_frontier.bucket(0).insert(remaining_vertices.begin(), remaining_vertices.end());

    if (loop_counter == 1) {
      // FIXME: The optimization below is not appropriate for the current
      // implementation since the neighbor with the highest priority
      // needs to be retrieved to update its rank if possible in the 'rank'
      // array. When using this primitive, it will stop once the first higher
      // priority neighbor is found which may not be the highest. This
      // will lead to more iterations until the highest priority neighbor
      // is found if any (this will increase the overall runtime).
      per_v_transform_reduce_if_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        multi_gpu ? dst_rank_cache.view()
                  : make_edge_dst_property_view<vertex_t, vertex_t>(
                      graph_view, ranks.begin(), ranks.size()),
        edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return dst_rank; },
        std::numeric_limits<vertex_t>::lowest(),
        reduce_op::any<vertex_t>(),
        // just use auto, auto remove src_rank and wt # FIXME: address this.
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return src < dst_rank; },
        max_outgoing_ranks.begin(),
        false); // FIXME: Set expensive check to False
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
        [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return dst_rank; },
        std::numeric_limits<vertex_t>::lowest(),
        cugraph::reduce_op::maximum<vertex_t>{},
        max_outgoing_ranks.begin(),
        false);
    }
  
    auto pair_idx_vertex_first = thrust::make_zip_iterator( // FIXME: rename this.
      thrust::make_counting_iterator<size_t>(0),
      remaining_vertices.begin()
      );
    

    //
    // If the max neighbor of a vertex is already in MIS (i.e. has rank
    // std::numeric_limits<vertex_t>::max()), discard it, otherwise,
    // include the vertex if it has larger rank than its maximum rank neighbor
    //

    // Use thrust::stable_partition to keep track of vertices that only needs to have
    // their property updated
    auto last = thrust::stable_partition(
      handle.get_thrust_policy(),
      pair_idx_vertex_first, // FIXME: Same here, no need to pass an iterator, just follow the pseudo code
      pair_idx_vertex_first + remaining_vertices.size(),
      [max_rank_neighbor_first = max_outgoing_ranks.begin(),
      ranks                   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
      v_first = graph_view.local_vertex_partition_range_first()] __device__(auto pair_vidx_v_priority) {

        auto vidx = thrust::get<0>(pair_vidx_v_priority);
        auto v = thrust::get<1>(pair_vidx_v_priority);
        auto v_offset          = v - v_first;
        auto max_neighbor_rank = *(max_rank_neighbor_first + vidx);
        auto rank_of_v         = ranks[v_offset];

        if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
          ranks[v_offset] = std::numeric_limits<vertex_t>::lowest();
          return false;
        }

        if (rank_of_v >= max_neighbor_rank) {
          ranks[v_offset] = std::numeric_limits<vertex_t>::max();
          return false;
        }
        return true;
      });

    nr_remaining_local_vertices_to_check = cuda::std::distance(pair_idx_vertex_first, last),
                          handle.get_stream();

    max_outgoing_ranks.resize(0, handle.get_stream());
    max_outgoing_ranks.shrink_to_fit(handle.get_stream());

    if (multi_gpu) {
      // FIXME: rename to 'nr_remaining_vertices_to_check' to
      // 'nr_remaining_global_vertices_to_check'
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_local_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    } else {
      nr_remaining_vertices_to_check = nr_remaining_local_vertices_to_check;
    }

    if (nr_remaining_vertices_to_check == 0) { break; }

  }
  
  vertex_t nr_vertices_included_in_mis = thrust::count_if(
    handle.get_thrust_policy(), ranks.begin(), ranks.end(), [] __device__(auto v_rank) {
      return v_rank >= std::numeric_limits<vertex_t>::max();
    });

  // Build MIS and return

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto start_build_mis = high_resolution_clock::now();

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

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> maximal_independent_set(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::random::RngState& rng_state)
{
  return detail::maximal_independent_set(handle, graph_view, rng_state);
}

}  // namespace cugraph

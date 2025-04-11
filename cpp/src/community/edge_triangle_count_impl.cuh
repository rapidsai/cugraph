/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/per_v_pair_dst_nbr_intersection.cuh"
#include "prims/transform_e.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/integer_utils.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct update_edges_p_r_q_r_num_triangles {
  size_t num_edges{};
  const edge_t edge_first_or_second{};
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<edge_t> num_triangles{};

  EdgeIterator edge_first{};

  __device__ void operator()(size_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = cuda::std::distance(intersection_offsets.begin() + 1, itr);
    if (edge_first_or_second == 0) {
      auto p_r_pair = thrust::make_tuple(thrust::get<0>(*(edge_first + chunk_start + idx)),
                                         intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_pair);

      assert(*itr_p_r_p_q == p_r_pair);
      idx = cuda::std::distance(edge_first, itr_p_r_p_q);
    } else {
      auto p_r_pair = thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)),
                                         intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_pair);
      assert(*itr_p_r_p_q == p_r_pair);
      idx = cuda::std::distance(edge_first, itr_p_r_p_q);
    }
    cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx]);
    auto r = atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct extract_p_r_q_r {
  size_t chunk_start{};
  size_t p_r_or_q_r{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  EdgeIterator edge_first;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = cuda::std::distance(intersection_offsets.begin() + 1, itr);

    if (p_r_or_q_r == 0) {
      return thrust::make_tuple(thrust::get<0>(*(edge_first + chunk_start + idx)),
                                intersection_indices[i]);
    } else {
      return thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)),
                                intersection_indices[i]);
    }
  }
};

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct extract_q_r {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  EdgeIterator edge_first;

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx  = cuda::std::distance(intersection_offsets.begin() + 1, itr);
    auto pair = thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)),
                                   intersection_indices[i]);

    return pair;
  }
};

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> edge_triangle_count_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  using weight_t = float;
  rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
  std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore, std::ignore) =
    decompress_to_edgelist<vertex_t, edge_t, weight_t, int32_t>(
      handle, graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  auto edge_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  size_t edges_to_intersect_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);

  auto num_chunks =
    raft::div_rounding_up_safe(edgelist_srcs.size(), edges_to_intersect_per_iteration);
  size_t prev_chunk_size   = 0;
  auto num_remaining_edges = edgelist_srcs.size();
  rmm::device_uvector<edge_t> num_triangles(edgelist_srcs.size(), handle.get_stream());

  // auto my_rank = handle.get_comms().get_rank();
  if constexpr (multi_gpu) {
    num_chunks = host_scalar_allreduce(
      handle.get_comms(), num_chunks, raft::comms::op_t::MAX, handle.get_stream());
  }

  // Need to ensure that the vector has its values initialized to 0 before incrementing
  thrust::fill(handle.get_thrust_policy(), num_triangles.begin(), num_triangles.end(), 0);

  for (size_t i = 0; i < num_chunks; ++i) {
    auto chunk_size = std::min(edges_to_intersect_per_iteration, num_remaining_edges);
    num_remaining_edges -= chunk_size;
    // Perform 'nbr_intersection' in chunks to reduce peak memory.
    auto [intersection_offsets, intersection_indices] =
      per_v_pair_dst_nbr_intersection(handle,
                                      graph_view,
                                      edge_first + prev_chunk_size,
                                      edge_first + prev_chunk_size + chunk_size,
                                      do_expensive_check);
    // Update the number of triangles of each (p, q) edges by looking at their intersection
    // size
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<edge_t>(0),
      thrust::make_counting_iterator<edge_t>(chunk_size),
      [chunk_start          = prev_chunk_size,
       num_triangles        = raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
       intersection_offsets = raft::device_span<size_t const>(
         intersection_offsets.data(), intersection_offsets.size())] __device__(auto i) {
        num_triangles[chunk_start + i] += (intersection_offsets[i + 1] - intersection_offsets[i]);
      });

    if constexpr (multi_gpu) {
      // stores all the pairs (p, r) and (q, r)
      auto vertex_pair_buffer_tmp = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        intersection_indices.size() * 2, handle.get_stream());

      // tabulate with the size of intersection_indices, and call binary search on
      // intersection_offsets to get (p, r).
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
        get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + intersection_indices.size(),
        extract_p_r_q_r<vertex_t, edge_t, decltype(edge_first)>{
          prev_chunk_size,
          0,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          edge_first});
      // FIXME: Consolidate both functions
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + intersection_indices.size(),
        get_dataframe_buffer_begin(vertex_pair_buffer_tmp) + (2 * intersection_indices.size()),
        extract_p_r_q_r<vertex_t, edge_t, decltype(edge_first)>{
          prev_chunk_size,
          1,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          edge_first});

      thrust::sort(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                   get_dataframe_buffer_end(vertex_pair_buffer_tmp));

      rmm::device_uvector<edge_t> increase_count_tmp(2 * intersection_indices.size(),
                                                     handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   increase_count_tmp.begin(),
                   increase_count_tmp.end(),
                   size_t{1});

      auto count_p_r_q_r = thrust::unique_count(handle.get_thrust_policy(),
                                                get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                                                get_dataframe_buffer_end(vertex_pair_buffer_tmp));

      rmm::device_uvector<edge_t> increase_count(count_p_r_q_r, handle.get_stream());

      auto vertex_pair_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        count_p_r_q_r, handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(vertex_pair_buffer_tmp),
                            get_dataframe_buffer_end(vertex_pair_buffer_tmp),
                            increase_count_tmp.begin(),
                            get_dataframe_buffer_begin(vertex_pair_buffer),
                            increase_count.begin(),
                            thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{});

      rmm::device_uvector<vertex_t> pair_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> pair_dsts(0, handle.get_stream());
      std::optional<rmm::device_uvector<edge_t>> pair_count{std::nullopt};

      std::optional<rmm::device_uvector<edge_t>> opt_increase_count =
        std::make_optional(rmm::device_uvector<edge_t>(increase_count.size(), handle.get_stream()));

      raft::copy<edge_t>((*opt_increase_count).begin(),
                         increase_count.begin(),
                         increase_count.size(),
                         handle.get_stream());

      // There are still multiple copies here but is it worth sorting and reducing again?
      std::tie(pair_srcs,
               pair_dsts,
               std::ignore,
               pair_count,
               std::ignore,
               std::ignore,
               std::ignore,
               std::ignore) =
        shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                               edge_t,
                                                                               weight_t,
                                                                               int32_t,
                                                                               int32_t>(
          handle,
          std::move(std::get<0>(vertex_pair_buffer)),
          std::move(std::get<1>(vertex_pair_buffer)),
          std::nullopt,
          // FIXME: Add general purpose function for shuffling vertex pairs and arbitrary attributes
          std::move(opt_increase_count),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          graph_view.vertex_partition_range_lasts());

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(pair_srcs.size()),
        [num_edges     = edgelist_srcs.size(),
         num_triangles = num_triangles.data(),
         pair_srcs     = pair_srcs.data(),
         pair_dsts     = pair_dsts.data(),
         pair_count    = (*pair_count).data(),
         edge_first] __device__(auto idx) {
          auto src          = pair_srcs[idx];
          auto dst          = pair_dsts[idx];
          auto p_r_q_r_pair = thrust::make_tuple(src, dst);

          // Find its position in 'edges'
          auto itr_p_r_q_r =
            thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_q_r_pair);

          assert(*itr_p_r_q_r == p_r_q_r_pair);
          auto idx_p_r_q_r = cuda::std::distance(edge_first, itr_p_r_q_r);

          cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(
            num_triangles[idx_p_r_q_r]);
          auto r = atomic_counter.fetch_add(pair_count[idx], cuda::std::memory_order_relaxed);
        });

    } else {
      // Given intersection offsets and indices that are used to update the number of
      // triangles of (p, q) edges where `r`s are the intersection indices, update
      // the number of triangles of the pairs (p, r) and (q, r).
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
        update_edges_p_r_q_r_num_triangles<vertex_t, edge_t, decltype(edge_first)>{
          edgelist_srcs.size(),
          0,
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
          edge_first});

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(intersection_indices.size()),
        update_edges_p_r_q_r_num_triangles<vertex_t, edge_t, decltype(edge_first)>{
          edgelist_srcs.size(),
          1,
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
          edge_first});
    }
    prev_chunk_size += chunk_size;
  }

  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> counts(
    handle, graph_view);

  cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> valid_edges(handle);
  valid_edges.insert(edgelist_srcs.begin(), edgelist_srcs.end(), edgelist_dsts.begin());

  auto cur_graph_view = graph_view;

  cugraph::transform_e(
    handle,
    graph_view,
    valid_edges,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    cugraph::edge_dummy_property_t{}.view(),
    [edge_first,
     edge_last     = edge_first + edgelist_srcs.size(),
     num_edges     = edgelist_srcs.size(),
     num_triangles = num_triangles.data()] __device__(auto src,
                                                      auto dst,
                                                      cuda::std::nullopt_t,
                                                      cuda::std::nullopt_t,
                                                      cuda::std::nullopt_t) {
      auto pair = thrust::make_tuple(src, dst);

      // Find its position in 'edges'
      auto itr_pair = thrust::lower_bound(thrust::seq, edge_first, edge_last, pair);
      auto idx_pair = cuda::std::distance(edge_first, itr_pair);
      return num_triangles[idx_pair];
    },
    counts.mutable_view(),
    false);

  return counts;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> edge_triangle_count(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  return detail::edge_triangle_count_impl(handle, graph_view, do_expensive_check);
}

}  // namespace cugraph

/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh"
#include "prims/transform_e.cuh"
#include "prims/edge_bucket.cuh"

#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

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
    auto idx = thrust::distance(intersection_offsets.begin() + 1, itr);
    if (edge_first_or_second == 0) {
      auto p_r_pair = thrust::make_tuple(thrust::get<0>(*(edge_first + chunk_start + idx)),
                                         intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_pair);

      assert(*itr_p_r_p_q == p_r_pair);
      idx = thrust::distance(edge_first, itr_p_r_p_q);
    } else {
      auto p_r_pair = thrust::make_tuple(thrust::get<1>(*(edge_first + chunk_start + idx)),
                                         intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq, edge_first, edge_first + num_edges, p_r_pair);
      assert(*itr_p_r_p_q == p_r_pair);
      idx = thrust::distance(edge_first, itr_p_r_p_q);
    }
    cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx]);
    auto r = atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<!multi_gpu, edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>> edge_triangle_count_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view)
{
  using weight_t = float;
  rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
  std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) = decompress_to_edgelist(
    handle,
    graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
    std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    std::optional<raft::device_span<vertex_t const>>(std::nullopt));

  auto edge_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());

  size_t approx_edges_to_intersect_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 17);

  auto num_chunks        = ((edgelist_srcs.size() % approx_edges_to_intersect_per_iteration) == 0)
                             ? (edgelist_srcs.size() / approx_edges_to_intersect_per_iteration)
                             : (edgelist_srcs.size() / approx_edges_to_intersect_per_iteration) + 1;
  size_t prev_chunk_size = 0;
  auto num_edges         = edgelist_srcs.size();
  rmm::device_uvector<edge_t> num_triangles(edgelist_srcs.size(), handle.get_stream());

  // Need to ensure that the vector has its values initialized to 0 before incrementing
  thrust::fill(handle.get_thrust_policy(), num_triangles.begin(), num_triangles.end(), 0);

  for (size_t i = 0; i < num_chunks; ++i) {
    auto chunk_size = std::min(approx_edges_to_intersect_per_iteration, num_edges);
    num_edges -= chunk_size;
    // Perform 'nbr_intersection' in chunks to reduce peak memory.
    auto [intersection_offsets, intersection_indices] =
      detail::nbr_intersection(handle,
                               graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               edge_first + prev_chunk_size,
                               edge_first + prev_chunk_size + chunk_size,
                               std::array<bool, 2>{true, true},
                               false /*FIXME: pass 'do_expensive_check' as argument*/);

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
        raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
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
        raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
        raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
        edge_first});

    prev_chunk_size += chunk_size;
  }

  std::vector<rmm::device_uvector<edge_t>> buffer{};
  buffer.push_back(std::move(num_triangles));
  buffer.reserve(num_triangles.size());
  buffer.push_back(std::move(num_triangles));

  auto buff_counts =
      edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>(
        std::move(buffer));

  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t> counts(handle, graph_view);

  cugraph::transform_e(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      buff_counts.view(),
      [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto count) {
        return count;
      },
      counts.mutable_view(),
      false);

  return counts;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>
edge_triangle_count(
  raft::handle_t const& handle, graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view)
{
  return detail::edge_triangle_count_impl(handle, graph_view);
}

}  // namespace cugraph

/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <detail/graph_partition_utils.cuh>

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename EdgeIterator>
struct update_edges_p_r_q_r_num_triangles {
  size_t num_edges{};  // rename to num_edges
  const edge_t edge_first_or_second{};
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
      auto p_r_pair =
        thrust::make_tuple(thrust::get<0>(*(edge_first + idx)), intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq,
                            edge_first,
                            edge_first + num_edges,  // pass the number of vertex pairs
                            p_r_pair);

      assert(*itr_p_r_p_q == p_r_pair);
      idx = thrust::distance(edge_first, itr_p_r_p_q);
    } else {
      auto p_r_pair =
        thrust::make_tuple(thrust::get<1>(*(edge_first + idx)), intersection_indices[i]);

      // Find its position in 'edges'
      auto itr_p_r_p_q =
        thrust::lower_bound(thrust::seq,
                            edge_first,
                            edge_first + num_edges,  // pass the number of vertex pairs
                            p_r_pair);
      assert(*itr_p_r_p_q == p_r_pair);
      idx = thrust::distance(edge_first, itr_p_r_p_q);
    }
    cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(num_triangles[idx]);
    auto r = atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<edge_t>> edge_triangle_count_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t> edgelist_srcs,
  raft::device_span<vertex_t> edgelist_dsts)
{
  auto edge_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());

  auto [intersection_offsets, intersection_indices] =
    detail::nbr_intersection(handle,
                             graph_view,
                             cugraph::edge_dummy_property_t{}.view(),
                             edge_first,
                             edge_first + edgelist_srcs.size(),
                             std::array<bool, 2>{true, true},
                             false /*FIXME: pass 'do_expensive_check' as argument*/);

  // FIXME: invalid type for unsigned integers.
  rmm::device_uvector<edge_t> num_triangles(edgelist_srcs.size(), handle.get_stream());

  // Update the number of triangles of each (p, q) edges by looking at their intersection
  // size
  thrust::adjacent_difference(handle.get_thrust_policy(),
                              intersection_offsets.begin() + 1,
                              intersection_offsets.end(),
                              num_triangles.begin());

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
      raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
      raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()),
      raft::device_span<edge_t>(num_triangles.data(), num_triangles.size()),
      edge_first});

  return num_triangles;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t> edge_triangle_count(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t> edgelist_srcs,
  raft::device_span<vertex_t> edgelist_dsts)
{
  return detail::edge_triangle_count_impl(handle, graph_view, edgelist_srcs, edgelist_dsts);
}

}  // namespace cugraph

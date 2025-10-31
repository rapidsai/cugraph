/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <optional>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename label_t, typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>>
remove_visited_vertices_from_frontier(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& frontier_vertices,
  std::optional<rmm::device_uvector<label_t>>&& frontier_vertex_labels,
  std::optional<rmm::device_uvector<edge_time_t>>&& frontier_vertex_times,
  raft::device_span<vertex_t const> vertices_used_as_source,
  std::optional<raft::device_span<label_t const>> vertex_labels_used_as_source)
{
  if (frontier_vertex_times) {
    if (frontier_vertex_labels) {
      auto sort_iter = thrust::make_zip_iterator(
        frontier_vertex_labels->begin(), frontier_vertices.begin(), frontier_vertex_times->begin());
      thrust::sort(handle.get_thrust_policy(), sort_iter, sort_iter + frontier_vertices.size());

      auto begin_iter =
        thrust::make_zip_iterator(frontier_vertex_labels->begin(), frontier_vertices.begin());
      auto new_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                           begin_iter,
                                           begin_iter + frontier_vertices.size(),
                                           frontier_vertex_times->begin(),
                                           begin_iter,
                                           frontier_vertex_times->begin(),
                                           thrust::equal_to<cuda::std::tuple<label_t, vertex_t>>{},
                                           thrust::minimum<edge_time_t>{});

      frontier_vertices.resize(cuda::std::distance(begin_iter, new_end.first), handle.get_stream());
      frontier_vertex_labels->resize(cuda::std::distance(begin_iter, new_end.first),
                                     handle.get_stream());
      frontier_vertex_times->resize(cuda::std::distance(begin_iter, new_end.first),
                                    handle.get_stream());
    } else {
      auto sort_iter =
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_times->begin());
      thrust::sort(handle.get_thrust_policy(), sort_iter, sort_iter + frontier_vertices.size());

      auto new_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                           frontier_vertices.begin(),
                                           frontier_vertices.end(),
                                           frontier_vertex_times->begin(),
                                           frontier_vertices.begin(),
                                           frontier_vertex_times->begin(),
                                           thrust::equal_to<vertex_t>{},
                                           thrust::minimum<edge_time_t>());

      frontier_vertices.resize(cuda::std::distance(frontier_vertices.begin(), new_end.first),
                               handle.get_stream());
      frontier_vertex_times->resize(
        cuda::std::distance(frontier_vertex_times->begin(), new_end.second), handle.get_stream());
    }
  } else {
    if (frontier_vertex_labels) {
      auto begin_iter =
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin());
      auto new_end = thrust::remove_if(
        handle.get_thrust_policy(),
        begin_iter,
        begin_iter + frontier_vertices.size(),
        begin_iter,
        [a_begin = vertices_used_as_source.begin(),
         a_end   = vertices_used_as_source.end(),
         b_begin = vertex_labels_used_as_source->begin(),
         b_end   = vertex_labels_used_as_source
                   ->end()] __device__(cuda::std::tuple<vertex_t, label_t> tuple) {
          return thrust::binary_search(thrust::seq,
                                       thrust::make_zip_iterator(a_begin, b_begin),
                                       thrust::make_zip_iterator(a_end, b_end),
                                       tuple);
        });

      frontier_vertices.resize(cuda::std::distance(begin_iter, new_end), handle.get_stream());
      frontier_vertex_labels->resize(cuda::std::distance(begin_iter, new_end), handle.get_stream());
    } else {
      auto new_end =
        thrust::copy_if(handle.get_thrust_policy(),
                        frontier_vertices.begin(),
                        frontier_vertices.end(),
                        frontier_vertices.begin(),
                        [a_begin = vertices_used_as_source.begin(),
                         a_end   = vertices_used_as_source.end()] __device__(vertex_t v) {
                          return !thrust::binary_search(thrust::seq, a_begin, a_end, v);
                        });
      frontier_vertices.resize(cuda::std::distance(frontier_vertices.begin(), new_end),
                               handle.get_stream());
    }
  }

  return std::make_tuple(std::move(frontier_vertices),
                         std::move(frontier_vertex_labels),
                         std::move(frontier_vertex_times));
}

}  // namespace detail
}  // namespace cugraph

/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/std/iterator>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

template <typename edge_t, typename VertexIterator>
rmm::device_uvector<edge_t> compute_sparse_offsets(
  VertexIterator edgelist_major_first,
  VertexIterator edgelist_major_last,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_last,
  bool edgelist_major_sorted,
  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<edge_t> offsets(static_cast<size_t>(major_range_last - major_range_first) + 1,
                                      stream_view);
  if (edgelist_major_sorted) {
    offsets.set_element_to_zero_async(0, stream_view);
    thrust::upper_bound(rmm::exec_policy(stream_view),
                        edgelist_major_first,
                        edgelist_major_last,
                        thrust::make_counting_iterator(major_range_first),
                        thrust::make_counting_iterator(major_range_last),
                        offsets.begin() + 1);
  } else {
    thrust::fill(rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), edge_t{0});

    auto offset_view = raft::device_span<edge_t>(offsets.data(), offsets.size());
    thrust::for_each(rmm::exec_policy(stream_view),
                     edgelist_major_first,
                     edgelist_major_last,
                     [offset_view, major_range_first] __device__(auto v) {
                       cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(
                         offset_view[v - major_range_first]);
                       atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
                     });

    thrust::exclusive_scan(
      rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), offsets.begin());
  }

  return offsets;
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>> compress_hypersparse_offsets(
  rmm::device_uvector<edge_t>&& offsets,
  vertex_t major_range_first,
  vertex_t major_hypersparse_first,
  vertex_t major_range_last,
  rmm::cuda_stream_view stream_view)
{
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  rmm::device_uvector<vertex_t> dcs_nzd_vertices =
    rmm::device_uvector<vertex_t>(major_range_last - major_hypersparse_first, stream_view);

  thrust::transform(rmm::exec_policy(stream_view),
                    thrust::make_counting_iterator(major_hypersparse_first),
                    thrust::make_counting_iterator(major_range_last),
                    dcs_nzd_vertices.begin(),
                    [major_range_first, offsets = offsets.data()] __device__(auto major) {
                      auto major_offset = major - major_range_first;
                      return offsets[major_offset + 1] - offsets[major_offset] > 0 ? major
                                                                                   : invalid_vertex;
                    });

  auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
    dcs_nzd_vertices.begin(), offsets.begin() + (major_hypersparse_first - major_range_first)));
  CUGRAPH_EXPECTS(
    dcs_nzd_vertices.size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "remove_if will fail (https://github.com/NVIDIA/thrust/issues/1302), work-around required.");
  dcs_nzd_vertices.resize(
    cuda::std::distance(pair_first,
                        thrust::remove_if(rmm::exec_policy(stream_view),
                                          pair_first,
                                          pair_first + dcs_nzd_vertices.size(),
                                          [] __device__(auto pair) {
                                            return thrust::get<0>(pair) == invalid_vertex;
                                          })),
    stream_view);
  dcs_nzd_vertices.shrink_to_fit(stream_view);
  if (static_cast<vertex_t>(dcs_nzd_vertices.size()) < major_range_last - major_hypersparse_first) {
    // copying offsets.back() to the new last position
    thrust::copy(
      rmm::exec_policy(stream_view),
      offsets.begin() + (major_range_last - major_range_first),
      offsets.end(),
      offsets.begin() + (major_hypersparse_first - major_range_first) + dcs_nzd_vertices.size());
    offsets.resize((major_hypersparse_first - major_range_first) + dcs_nzd_vertices.size() + 1,
                   stream_view);
    offsets.shrink_to_fit(stream_view);
  }

  return std::make_tuple(std::move(offsets), std::move(dcs_nzd_vertices));
}

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <typename vertex_t, typename edge_t, typename edge_value_t, bool store_transposed>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           decltype(allocate_dataframe_buffer<edge_value_t>(size_t{0}, rmm::cuda_stream_view{})),
           std::optional<rmm::device_uvector<vertex_t>>>
sort_and_compress_edgelist(
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  decltype(allocate_dataframe_buffer<edge_value_t>(0, rmm::cuda_stream_view{}))&& edgelist_values,
  vertex_t major_range_first,
  std::optional<vertex_t> major_hypersparse_first,
  vertex_t major_range_last,
  vertex_t /* minor_range_first */,
  vertex_t /* minor_range_last */,
  size_t mem_frugal_threshold,
  rmm::cuda_stream_view stream_view)
{
  auto edgelist_majors = std::move(store_transposed ? edgelist_dsts : edgelist_srcs);
  auto edgelist_minors = std::move(store_transposed ? edgelist_srcs : edgelist_dsts);

  rmm::device_uvector<edge_t> offsets(0, stream_view);
  rmm::device_uvector<vertex_t> indices(0, stream_view);
  auto values     = allocate_dataframe_buffer<edge_value_t>(0, stream_view);
  auto pair_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
  if (edgelist_minors.size() > mem_frugal_threshold) {
    offsets = compute_sparse_offsets<edge_t>(edgelist_majors.begin(),
                                             edgelist_majors.end(),
                                             major_range_first,
                                             major_range_last,
                                             false,
                                             stream_view);

    auto pivot = major_range_first + static_cast<vertex_t>(cuda::std::distance(
                                       offsets.begin(),
                                       thrust::lower_bound(rmm::exec_policy(stream_view),
                                                           offsets.begin(),
                                                           offsets.end(),
                                                           edgelist_minors.size() / 2)));
    auto second_first =
      detail::mem_frugal_partition(pair_first,
                                   pair_first + edgelist_minors.size(),
                                   get_dataframe_buffer_begin(edgelist_values),
                                   thrust_tuple_get<thrust::tuple<vertex_t, vertex_t>, 0>{},
                                   pivot,
                                   stream_view);
    thrust::sort_by_key(rmm::exec_policy(stream_view),
                        pair_first,
                        std::get<0>(second_first),
                        get_dataframe_buffer_begin(edgelist_values));
    thrust::sort_by_key(rmm::exec_policy(stream_view),
                        std::get<0>(second_first),
                        pair_first + edgelist_minors.size(),
                        std::get<1>(second_first));
  } else {
    thrust::sort_by_key(rmm::exec_policy(stream_view),
                        pair_first,
                        pair_first + edgelist_minors.size(),
                        get_dataframe_buffer_begin(edgelist_values));

    offsets = compute_sparse_offsets<edge_t>(edgelist_majors.begin(),
                                             edgelist_majors.end(),
                                             major_range_first,
                                             major_range_last,
                                             true,
                                             stream_view);
  }
  indices = std::move(edgelist_minors);
  values  = std::move(edgelist_values);

  edgelist_majors.resize(0, stream_view);
  edgelist_majors.shrink_to_fit(stream_view);

  std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
  if (major_hypersparse_first) {
    std::tie(offsets, dcs_nzd_vertices) = compress_hypersparse_offsets(std::move(offsets),
                                                                       major_range_first,
                                                                       *major_hypersparse_first,
                                                                       major_range_last,
                                                                       stream_view);
  }

  return std::make_tuple(
    std::move(offsets), std::move(indices), std::move(values), std::move(dcs_nzd_vertices));
}

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <typename vertex_t, typename edge_t, bool store_transposed>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<vertex_t>>>
sort_and_compress_edgelist(rmm::device_uvector<vertex_t>&& edgelist_srcs,
                           rmm::device_uvector<vertex_t>&& edgelist_dsts,
                           vertex_t major_range_first,
                           std::optional<vertex_t> major_hypersparse_first,
                           vertex_t major_range_last,
                           vertex_t /* minor_range_first */,
                           vertex_t /* minor_range_last */,
                           size_t mem_frugal_threshold,
                           rmm::cuda_stream_view stream_view)
{
  auto edgelist_majors = std::move(store_transposed ? edgelist_dsts : edgelist_srcs);
  auto edgelist_minors = std::move(store_transposed ? edgelist_srcs : edgelist_dsts);

  rmm::device_uvector<edge_t> offsets(0, stream_view);
  rmm::device_uvector<vertex_t> indices(0, stream_view);
  if (edgelist_minors.size() > mem_frugal_threshold) {
    static_assert((sizeof(vertex_t) == 4) || (sizeof(vertex_t) == 8));
    if ((sizeof(vertex_t) == 8) && (static_cast<size_t>(major_range_last - major_range_first) <=
                                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()))) {
      rmm::device_uvector<uint32_t> edgelist_major_offsets(edgelist_majors.size(), stream_view);
      thrust::transform(
        rmm::exec_policy_nosync(stream_view),
        edgelist_majors.begin(),
        edgelist_majors.end(),
        edgelist_major_offsets.begin(),
        cuda::proclaim_return_type<uint32_t>([major_range_first] __device__(vertex_t major) {
          return static_cast<uint32_t>(major - major_range_first);
        }));
      edgelist_majors.resize(0, stream_view);
      edgelist_majors.shrink_to_fit(stream_view);

      offsets =
        compute_sparse_offsets<edge_t>(edgelist_major_offsets.begin(),
                                       edgelist_major_offsets.end(),
                                       uint32_t{0},
                                       static_cast<uint32_t>(major_range_last - major_range_first),
                                       false,
                                       stream_view);
      std::array<uint32_t, 3> pivots{};
      for (size_t i = 0; i < 3; ++i) {
        pivots[i] = static_cast<uint32_t>(cuda::std::distance(
          offsets.begin(),
          thrust::lower_bound(rmm::exec_policy(stream_view),
                              offsets.begin(),
                              offsets.end(),
                              static_cast<edge_t>((edgelist_major_offsets.size() * (i + 1)) / 4))));
      }

      auto pair_first =
        thrust::make_zip_iterator(edgelist_major_offsets.begin(), edgelist_minors.begin());
      auto second_half_first =
        detail::mem_frugal_partition(pair_first,
                                     pair_first + edgelist_major_offsets.size(),
                                     thrust_tuple_get<thrust::tuple<uint32_t, vertex_t>, 0>{},
                                     pivots[1],
                                     stream_view);
      auto second_quarter_first =
        detail::mem_frugal_partition(pair_first,
                                     second_half_first,
                                     thrust_tuple_get<thrust::tuple<uint32_t, vertex_t>, 0>{},
                                     pivots[0],
                                     stream_view);
      auto last_quarter_first =
        detail::mem_frugal_partition(second_half_first,
                                     pair_first + edgelist_major_offsets.size(),
                                     thrust_tuple_get<thrust::tuple<uint32_t, vertex_t>, 0>{},
                                     pivots[2],
                                     stream_view);
      thrust::sort(rmm::exec_policy(stream_view), pair_first, second_quarter_first);
      thrust::sort(rmm::exec_policy(stream_view), second_quarter_first, second_half_first);
      thrust::sort(rmm::exec_policy(stream_view), second_half_first, last_quarter_first);
      thrust::sort(rmm::exec_policy(stream_view),
                   last_quarter_first,
                   pair_first + edgelist_major_offsets.size());
    } else {
      offsets = compute_sparse_offsets<edge_t>(edgelist_majors.begin(),
                                               edgelist_majors.end(),
                                               major_range_first,
                                               major_range_last,
                                               false,
                                               stream_view);
      std::array<vertex_t, 3> pivots{};
      for (size_t i = 0; i < 3; ++i) {
        pivots[i] =
          major_range_first +
          static_cast<vertex_t>(cuda::std::distance(
            offsets.begin(),
            thrust::lower_bound(rmm::exec_policy(stream_view),
                                offsets.begin(),
                                offsets.end(),
                                static_cast<edge_t>((edgelist_minors.size() * (i + 1)) / 4))));
      }
      auto edge_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
      auto second_half_first =
        detail::mem_frugal_partition(edge_first,
                                     edge_first + edgelist_majors.size(),
                                     thrust_tuple_get<thrust::tuple<vertex_t, vertex_t>, 0>{},
                                     pivots[1],
                                     stream_view);
      auto second_quarter_first =
        detail::mem_frugal_partition(edge_first,
                                     second_half_first,
                                     thrust_tuple_get<thrust::tuple<vertex_t, vertex_t>, 0>{},
                                     pivots[0],
                                     stream_view);
      auto last_quarter_first =
        detail::mem_frugal_partition(second_half_first,
                                     edge_first + edgelist_majors.size(),
                                     thrust_tuple_get<thrust::tuple<vertex_t, vertex_t>, 0>{},
                                     pivots[2],
                                     stream_view);
      thrust::sort(rmm::exec_policy(stream_view), edge_first, second_quarter_first);
      thrust::sort(rmm::exec_policy(stream_view), second_quarter_first, second_half_first);
      thrust::sort(rmm::exec_policy(stream_view), second_half_first, last_quarter_first);
      thrust::sort(
        rmm::exec_policy(stream_view), last_quarter_first, edge_first + edgelist_majors.size());
      edgelist_majors.resize(0, stream_view);
      edgelist_majors.shrink_to_fit(stream_view);
    }
  } else {
    auto edge_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
    thrust::sort(rmm::exec_policy(stream_view), edge_first, edge_first + edgelist_minors.size());
    offsets = compute_sparse_offsets<edge_t>(edgelist_majors.begin(),
                                             edgelist_majors.end(),
                                             major_range_first,
                                             major_range_last,
                                             true,
                                             stream_view);
    edgelist_majors.resize(0, stream_view);
    edgelist_majors.shrink_to_fit(stream_view);
  }
  indices = std::move(edgelist_minors);

  std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
  if (major_hypersparse_first) {
    std::tie(offsets, dcs_nzd_vertices) = compress_hypersparse_offsets(std::move(offsets),
                                                                       major_range_first,
                                                                       *major_hypersparse_first,
                                                                       major_range_last,
                                                                       stream_view);
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(dcs_nzd_vertices));
}

template <typename edge_t, typename VertexIterator, typename EdgeValueIterator>
void sort_adjacency_list(raft::handle_t const& handle,
                         raft::device_span<edge_t const> offsets,
                         VertexIterator index_first /* [INOUT] */,
                         VertexIterator index_last /* [INOUT] */,
                         EdgeValueIterator edge_value_first /* [INOUT] */)
{
  using vertex_t     = typename thrust::iterator_traits<VertexIterator>::value_type;
  using edge_value_t = typename thrust::iterator_traits<EdgeValueIterator>::value_type;

  // 1. Check if there is anything to sort

  auto num_edges = static_cast<edge_t>(cuda::std::distance(index_first, index_last));
  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
    handle, offsets, num_edges, approx_edges_to_sort_per_iteration);
  auto num_chunks = h_vertex_offsets.size() - 1;

  // 3. Segmented sort each vertex's neighbors

  size_t max_chunk_size{0};
  for (size_t i = 0; i < num_chunks; ++i) {
    max_chunk_size =
      std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[i + 1] - h_edge_offsets[i]));
  }
  rmm::device_uvector<vertex_t> segment_sorted_indices(max_chunk_size, handle.get_stream());
  rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
  auto segment_sorted_values =
    allocate_dataframe_buffer<edge_value_t>(max_chunk_size, handle.get_stream());
  if constexpr (std::is_arithmetic_v<edge_value_t>) {
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};
      auto offset_first = thrust::make_transform_iterator(offsets.data() + h_vertex_offsets[i],
                                                          shift_left_t<edge_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          edge_value_first + h_edge_offsets[i],
                                          segment_sorted_values.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          edge_value_first + h_edge_offsets[i],
                                          segment_sorted_values.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   segment_sorted_indices.begin(),
                   segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   index_first + h_edge_offsets[i]);
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(segment_sorted_values),
                   get_dataframe_buffer_begin(segment_sorted_values) +
                     (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   edge_value_first + h_edge_offsets[i]);
    }
  } else {  // cub's segmented sort does not support thrust iterators (so we can't directly sort
            // edge values with thrust::zip_iterator)
    rmm::device_uvector<edge_t> input_edge_value_offsets(max_chunk_size, handle.get_stream());
    rmm::device_uvector<edge_t> segment_sorted_edge_value_offsets(max_chunk_size,
                                                                  handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     input_edge_value_offsets.begin(),
                     input_edge_value_offsets.end(),
                     edge_t{0});
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};
      auto offset_first = thrust::make_transform_iterator(offsets.data() + h_vertex_offsets[i],
                                                          shift_left_t<edge_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          input_edge_value_offsets.data(),
                                          segment_sorted_edge_value_offsets.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          index_first + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          input_edge_value_offsets.data(),
                                          segment_sorted_edge_value_offsets.data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   segment_sorted_indices.begin(),
                   segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   index_first + h_edge_offsets[i]);
      thrust::gather(
        handle.get_thrust_policy(),
        segment_sorted_edge_value_offsets.begin(),
        segment_sorted_edge_value_offsets.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
        edge_value_first + h_edge_offsets[i],
        get_dataframe_buffer_begin(segment_sorted_values));
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(segment_sorted_values),
                   get_dataframe_buffer_begin(segment_sorted_values) +
                     (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   edge_value_first + h_edge_offsets[i]);
    }
  }
}

template <typename edge_t, typename VertexIterator>
void sort_adjacency_list(raft::handle_t const& handle,
                         raft::device_span<edge_t const> offsets,
                         VertexIterator index_first /* [INOUT] */,
                         VertexIterator index_last /* [INOUT] */)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;

  // 1. Check if there is anything to sort

  auto num_edges = static_cast<edge_t>(cuda::std::distance(index_first, index_last));
  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
    handle, offsets, num_edges, approx_edges_to_sort_per_iteration);
  auto num_chunks = h_vertex_offsets.size() - 1;

  // 3. Segmented sort each vertex's neighbors

  size_t max_chunk_size{0};
  for (size_t i = 0; i < num_chunks; ++i) {
    max_chunk_size =
      std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[i + 1] - h_edge_offsets[i]));
  }
  rmm::device_uvector<vertex_t> segment_sorted_indices(max_chunk_size, handle.get_stream());
  rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t tmp_storage_bytes{0};
    auto offset_first = thrust::make_transform_iterator(offsets.data() + h_vertex_offsets[i],
                                                        shift_left_t<edge_t>{h_edge_offsets[i]});
    cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                       tmp_storage_bytes,
                                       index_first + h_edge_offsets[i],
                                       segment_sorted_indices.data(),
                                       h_edge_offsets[i + 1] - h_edge_offsets[i],
                                       h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                       offset_first,
                                       offset_first + 1,
                                       handle.get_stream());
    if (tmp_storage_bytes > d_tmp_storage.size()) {
      d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
    }
    cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                       tmp_storage_bytes,
                                       index_first + h_edge_offsets[i],
                                       segment_sorted_indices.data(),
                                       h_edge_offsets[i + 1] - h_edge_offsets[i],
                                       h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                       offset_first,
                                       offset_first + 1,
                                       handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 segment_sorted_indices.begin(),
                 segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                 index_first + h_edge_offsets[i]);
  }
}

template <typename comparison_t>
std::tuple<size_t, rmm::device_uvector<uint32_t>> mark_entries(raft::handle_t const& handle,
                                                               size_t num_entries,
                                                               comparison_t comparison)
{
  rmm::device_uvector<uint32_t> marked_entries(cugraph::packed_bool_size(num_entries),
                                               handle.get_stream());

  thrust::tabulate(handle.get_thrust_policy(),
                   marked_entries.begin(),
                   marked_entries.end(),
                   [comparison, num_entries] __device__(size_t idx) {
                     auto word          = cugraph::packed_bool_empty_mask();
                     size_t start_index = idx * cugraph::packed_bools_per_word();
                     size_t bits_in_this_word =
                       (start_index + cugraph::packed_bools_per_word() < num_entries)
                         ? cugraph::packed_bools_per_word()
                         : (num_entries - start_index);

                     for (size_t bit = 0; bit < bits_in_this_word; ++bit) {
                       if (comparison(start_index + bit)) word |= cugraph::packed_bool_mask(bit);
                     }

                     return word;
                   });

  size_t bit_count = detail::count_set_bits(handle, marked_entries.begin(), num_entries);

  return std::make_tuple(bit_count, std::move(marked_entries));
}

template <typename T>
rmm::device_uvector<T> keep_flagged_elements(raft::handle_t const& handle,
                                             rmm::device_uvector<T>&& vector,
                                             raft::device_span<uint32_t const> keep_flags,
                                             size_t keep_count)
{
  rmm::device_uvector<T> result(keep_count, handle.get_stream());

  detail::copy_if_mask_set(
    handle, vector.begin(), vector.end(), keep_flags.begin(), result.begin());
  vector.resize(0, handle.get_stream());
  vector.shrink_to_fit(handle.get_stream());

  return result;
}

}  // namespace detail
}  // namespace cugraph

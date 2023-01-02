/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
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

template <bool store_transposed,
          typename vertex_t,
          typename edge_t,
          typename EdgeIterator,
          typename EdgeValueIterator>
struct update_edge_t {
  raft::device_span<edge_t> offsets{};
  raft::device_span<vertex_t> indices{};
  EdgeValueIterator edge_value_first{};
  vertex_t major_range_first{};

  __device__ void operator()(typename thrust::iterator_traits<EdgeIterator>::value_type e) const
  {
    auto s      = thrust::get<0>(e);
    auto d      = thrust::get<1>(e);
    auto major  = store_transposed ? d : s;
    auto minor  = store_transposed ? s : d;
    auto start  = offsets[major - major_range_first];
    auto degree = offsets[(major - major_range_first) + 1] - start;
    auto idx =
      atomicAdd(&indices[start + degree - 1], vertex_t{1});  // use the last element as a counter
    // FIXME: we can actually store minor - minor_range_first instead of minor to save memory if
    // minor can be larger than 32 bit but minor -  minor_range_first fits within 32 bit
    indices[start + idx] = minor;  // overwrite the counter only if idx == degree - 1 (no race)
    if constexpr (!std::is_same_v<EdgeValueIterator, void*>) {
      auto value                          = thrust::get<2>(e);
      *(edge_value_first + (start + idx)) = value;
    }
  }
};

template <typename edge_t>
struct rebase_offset_t {
  edge_t base_offset{};
  __device__ edge_t operator()(edge_t offset) const { return offset - base_offset; }
};

template <typename idx_t, typename offset_t>
rmm::device_uvector<idx_t> expand_sparse_offsets(raft::device_span<offset_t const> offsets,
                                                 idx_t base_idx,
                                                 rmm::cuda_stream_view stream_view)
{
  assert(offsets.size() > 0);

  offset_t num_entries{0};
  raft::update_host(&num_entries, offsets.data() + offsets.size() - 1, 1, stream_view);

  rmm::device_uvector<idx_t> results(num_entries, stream_view);

  if (num_entries > 0) {
    thrust::fill(rmm::exec_policy(stream_view), results.begin(), results.end(), idx_t{0});

    raft::update_device(results.data(), &base_idx, 1, stream_view);

    thrust::for_each(
      rmm::exec_policy(stream_view),
      offsets.begin() + 1,
      offsets.end(),
      [d_results = results.data(), n_results = results.size()] __device__(auto offset) {
        if (offset < n_results) { atomicAdd(&d_results[offset], idx_t{1}); }
      });
    thrust::inclusive_scan(
      rmm::exec_policy(stream_view), results.begin(), results.end(), results.begin());
  }

  return results;
}

template <typename edge_t, typename VertexIterator>
rmm::device_uvector<edge_t> compute_sparse_offsets(
  VertexIterator edgelist_major_first,
  VertexIterator edgelist_major_last,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_last,
  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<edge_t> offsets((major_range_last - major_range_first) + 1, stream_view);
  thrust::fill(rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), edge_t{0});

  auto offset_view = raft::device_span<edge_t>(offsets.data(), offsets.size());
  thrust::for_each(rmm::exec_policy(stream_view),
                   edgelist_major_first,
                   edgelist_major_last,
                   [offset_view, major_range_first] __device__(auto v) {
                     atomicAdd(&offset_view[v - major_range_first], edge_t{1});
                   });
  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), offsets.begin());

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
  dcs_nzd_vertices.resize(thrust::distance(pair_first,
                                           thrust::remove_if(rmm::exec_policy(stream_view),
                                                             pair_first,
                                                             pair_first + dcs_nzd_vertices.size(),
                                                             [] __device__(auto pair) {
                                                               return thrust::get<0>(pair) ==
                                                                      invalid_vertex;
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
template <typename edge_t,
          bool store_transposed,
          typename VertexIterator,
          typename EdgeValueIterator>
std::tuple<
  rmm::device_uvector<edge_t>,
  rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type>,
  decltype(
    allocate_dataframe_buffer<typename thrust::iterator_traits<EdgeValueIterator>::value_type>(
      size_t{0}, rmm::cuda_stream_view{})),
  std::optional<rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type>>>
compress_edgelist(
  VertexIterator edgelist_src_first,
  VertexIterator edgelist_src_last,
  VertexIterator edgelist_dst_first,
  EdgeValueIterator edge_value_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_first,
  std::optional<typename thrust::iterator_traits<VertexIterator>::value_type>
    major_hypersparse_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_last,
  typename thrust::iterator_traits<VertexIterator>::value_type /* minor_range_first */,
  typename thrust::iterator_traits<VertexIterator>::value_type /* minor_range_last */,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = std::remove_cv_t<typename thrust::iterator_traits<VertexIterator>::value_type>;
  using edge_value_t =
    std::remove_cv_t<typename thrust::iterator_traits<EdgeValueIterator>::value_type>;

  auto number_of_edges =
    static_cast<edge_t>(thrust::distance(edgelist_src_first, edgelist_src_last));

  auto offsets = compute_sparse_offsets<edge_t>(
    store_transposed ? edgelist_dst_first : edgelist_src_first,
    store_transposed ? edgelist_dst_first + number_of_edges : edgelist_src_last,
    major_range_first,
    major_range_last,
    stream_view);

  rmm::device_uvector<vertex_t> indices(number_of_edges, stream_view);
  thrust::fill(rmm::exec_policy(stream_view), indices.begin(), indices.end(), vertex_t{0});
  auto values = allocate_dataframe_buffer<edge_value_t>(number_of_edges, stream_view);

  auto offset_view = raft::device_span<edge_t>(offsets.data(), offsets.size());
  auto index_view  = raft::device_span<vertex_t>(indices.data(), indices.size());
  auto edge_first  = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_src_first, edgelist_dst_first, edge_value_first));
  thrust::for_each(
    rmm::exec_policy(stream_view),
    edge_first,
    edge_first + number_of_edges,
    update_edge_t<store_transposed,
                  vertex_t,
                  edge_t,
                  decltype(edge_first),
                  decltype(get_dataframe_buffer_begin(values))>{
      offset_view, index_view, get_dataframe_buffer_begin(values), major_range_first});

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
template <typename edge_t, bool store_transposed, typename VertexIterator>
std::tuple<
  rmm::device_uvector<edge_t>,
  rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type>,
  std::optional<rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type>>>
compress_edgelist(
  VertexIterator edgelist_src_first,
  VertexIterator edgelist_src_last,
  VertexIterator edgelist_dst_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_first,
  std::optional<typename thrust::iterator_traits<VertexIterator>::value_type>
    major_hypersparse_first,
  typename thrust::iterator_traits<VertexIterator>::value_type major_range_last,
  typename thrust::iterator_traits<VertexIterator>::value_type /* minor_range_first */,
  typename thrust::iterator_traits<VertexIterator>::value_type /* minor_range_last */,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = std::remove_cv_t<typename thrust::iterator_traits<VertexIterator>::value_type>;

  auto number_of_edges =
    static_cast<edge_t>(thrust::distance(edgelist_src_first, edgelist_src_last));

  auto offsets = compute_sparse_offsets<edge_t>(
    store_transposed ? edgelist_dst_first : edgelist_src_first,
    store_transposed ? edgelist_dst_first + number_of_edges : edgelist_src_last,
    major_range_first,
    major_range_last,
    stream_view);

  rmm::device_uvector<vertex_t> indices(number_of_edges, stream_view);
  thrust::fill(rmm::exec_policy(stream_view), indices.begin(), indices.end(), vertex_t{0});

  auto offset_view = raft::device_span<edge_t>(offsets.data(), offsets.size());
  auto index_view  = raft::device_span<vertex_t>(indices.data(), indices.size());
  auto edge_first =
    thrust::make_zip_iterator(thrust::make_tuple(edgelist_src_first, edgelist_dst_first));
  thrust::for_each(rmm::exec_policy(stream_view),
                   edge_first,
                   edge_first + number_of_edges,
                   update_edge_t<store_transposed, vertex_t, edge_t, decltype(edge_first), void*>{
                     offset_view, index_view, static_cast<void*>(nullptr), major_range_first});

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

  auto num_edges = static_cast<edge_t>(thrust::distance(index_first, index_last));
  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] =
    detail::compute_offset_aligned_edge_chunks(handle,
                                               offsets.data(),
                                               static_cast<vertex_t>(offsets.size() - 1),
                                               num_edges,
                                               approx_edges_to_sort_per_iteration);
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
      auto offset_first = thrust::make_transform_iterator(
        offsets.data() + h_vertex_offsets[i], rebase_offset_t<edge_t>{h_edge_offsets[i]});
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
      auto offset_first = thrust::make_transform_iterator(
        offsets.data() + h_vertex_offsets[i], rebase_offset_t<edge_t>{h_edge_offsets[i]});
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

  auto num_edges = static_cast<edge_t>(thrust::distance(index_first, index_last));
  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] =
    detail::compute_offset_aligned_edge_chunks(handle,
                                               offsets.data(),
                                               static_cast<vertex_t>(offsets.size() - 1),
                                               num_edges,
                                               approx_edges_to_sort_per_iteration);
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
                                                        rebase_offset_t<edge_t>{h_edge_offsets[i]});
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

}  // namespace detail

}  // namespace cugraph

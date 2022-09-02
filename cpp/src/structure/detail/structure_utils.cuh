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
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
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
#include <thrust/transform.h>

#include <algorithm>
#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

template <typename edge_t>
struct rebase_offset_t {
  edge_t base_offset{};
  __device__ edge_t operator()(edge_t offset) const { return offset - base_offset; }
};

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
compress_edgelist(edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
                  vertex_t major_range_first,
                  std::optional<vertex_t> major_hypersparse_first,
                  vertex_t major_range_last,
                  vertex_t /* minor_range_first */,
                  vertex_t /* minor_range_last */,
                  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<edge_t> offsets((major_range_last - major_range_first) + 1, stream_view);
  rmm::device_uvector<vertex_t> indices(edgelist.srcs.size(), stream_view);
  auto weights = edgelist.weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                      (*(edgelist.weights)).size(), stream_view)
                                  : std::nullopt;
  thrust::fill(rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), edge_t{0});
  thrust::fill(rmm::exec_policy(stream_view), indices.begin(), indices.end(), vertex_t{0});

  auto p_offsets = offsets.data();
  thrust::for_each(rmm::exec_policy(stream_view),
                   store_transposed ? edgelist.dsts.begin() : edgelist.srcs.begin(),
                   store_transposed ? edgelist.dsts.end() : edgelist.srcs.end(),
                   [p_offsets, major_range_first] __device__(auto v) {
                     atomicAdd(p_offsets + (v - major_range_first), edge_t{1});
                   });
  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), offsets.begin());

  auto p_indices = indices.data();
  if (edgelist.weights) {
    auto p_weights = (*weights).data();

    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      edgelist.srcs.begin(), edgelist.dsts.begin(), (*(edgelist.weights)).begin()));
    thrust::for_each(rmm::exec_policy(stream_view),
                     edge_first,
                     edge_first + edgelist.srcs.size(),
                     [p_offsets, p_indices, p_weights, major_range_first] __device__(auto e) {
                       auto s      = thrust::get<0>(e);
                       auto d      = thrust::get<1>(e);
                       auto w      = thrust::get<2>(e);
                       auto major  = store_transposed ? d : s;
                       auto minor  = store_transposed ? s : d;
                       auto start  = p_offsets[major - major_range_first];
                       auto degree = p_offsets[(major - major_range_first) + 1] - start;
                       auto idx    = atomicAdd(p_indices + (start + degree - 1),
                                            vertex_t{1});  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_range_first instead of minor to
                       // save memory if minor can be larger than 32 bit but minor -
                       // minor_range_first fits within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                       p_weights[start + idx] = w;
                     });
  } else {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(edgelist.srcs.begin(), edgelist.dsts.begin()));
    thrust::for_each(rmm::exec_policy(stream_view),
                     edge_first,
                     edge_first + edgelist.srcs.size(),
                     [p_offsets, p_indices, major_range_first] __device__(auto e) {
                       auto s      = thrust::get<0>(e);
                       auto d      = thrust::get<1>(e);
                       auto major  = store_transposed ? d : s;
                       auto minor  = store_transposed ? s : d;
                       auto start  = p_offsets[major - major_range_first];
                       auto degree = p_offsets[(major - major_range_first) + 1] - start;
                       auto idx    = atomicAdd(p_indices + (start + degree - 1),
                                            vertex_t{1});  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_range_first instead of minor to
                       // save memory if minor can be larger than 32 bit but minor -
                       // minor_range_first fits within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                     });
  }

  auto dcs_nzd_vertices = major_hypersparse_first
                            ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                major_range_last - *major_hypersparse_first, stream_view)
                            : std::nullopt;
  if (dcs_nzd_vertices) {
    auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

    thrust::transform(
      rmm::exec_policy(stream_view),
      thrust::make_counting_iterator(*major_hypersparse_first),
      thrust::make_counting_iterator(major_range_last),
      (*dcs_nzd_vertices).begin(),
      [major_range_first, offsets = offsets.data()] __device__(auto major) {
        auto major_offset = major - major_range_first;
        return offsets[major_offset + 1] - offsets[major_offset] > 0 ? major : invalid_vertex;
      });

    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple((*dcs_nzd_vertices).begin(),
                         offsets.begin() + (*major_hypersparse_first - major_range_first)));
    CUGRAPH_EXPECTS(
      (*dcs_nzd_vertices).size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "remove_if will fail (https://github.com/NVIDIA/thrust/issues/1302), work-around required.");
    (*dcs_nzd_vertices)
      .resize(thrust::distance(pair_first,
                               thrust::remove_if(rmm::exec_policy(stream_view),
                                                 pair_first,
                                                 pair_first + (*dcs_nzd_vertices).size(),
                                                 [] __device__(auto pair) {
                                                   return thrust::get<0>(pair) == invalid_vertex;
                                                 })),
              stream_view);
    (*dcs_nzd_vertices).shrink_to_fit(stream_view);
    if (static_cast<vertex_t>((*dcs_nzd_vertices).size()) <
        major_range_last - *major_hypersparse_first) {
      thrust::copy(rmm::exec_policy(stream_view),
                   offsets.begin() + (major_range_last - major_range_first),
                   offsets.end(),
                   offsets.begin() + (*major_hypersparse_first - major_range_first) +
                     (*dcs_nzd_vertices).size());
      offsets.resize(
        (*major_hypersparse_first - major_range_first) + (*dcs_nzd_vertices).size() + 1,
        stream_view);
      offsets.shrink_to_fit(stream_view);
    }
  }

  return std::make_tuple(
    std::move(offsets), std::move(indices), std::move(weights), std::move(dcs_nzd_vertices));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sort_adjacency_list(raft::handle_t const& handle,
                         edge_t const* offsets,
                         vertex_t* indices /* [INOUT] */,
                         std::optional<weight_t*> weights /* [INOUT] */,
                         vertex_t num_vertices,
                         edge_t num_edges)
{
  // 1. Check if there is anything to sort

  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_edge_chunks(
    handle, offsets, num_vertices, num_edges, approx_edges_to_sort_per_iteration);
  auto num_chunks = h_vertex_offsets.size() - 1;

  // 3. Segmented sort each vertex's neighbors

  size_t max_chunk_size{0};
  for (size_t i = 0; i < num_chunks; ++i) {
    max_chunk_size =
      std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[i + 1] - h_edge_offsets[i]));
  }
  rmm::device_uvector<vertex_t> segment_sorted_indices(max_chunk_size, handle.get_stream());
  auto segment_sorted_weights =
    weights ? std::make_optional<rmm::device_uvector<weight_t>>(max_chunk_size, handle.get_stream())
            : std::nullopt;
  rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t tmp_storage_bytes{0};
    auto offset_first = thrust::make_transform_iterator(offsets + h_vertex_offsets[i],
                                                        rebase_offset_t<edge_t>{h_edge_offsets[i]});
    if (weights) {
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          indices + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          (*weights) + h_edge_offsets[i],
                                          (*segment_sorted_weights).data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
    } else {
      cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                         tmp_storage_bytes,
                                         indices + h_edge_offsets[i],
                                         segment_sorted_indices.data(),
                                         h_edge_offsets[i + 1] - h_edge_offsets[i],
                                         h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                         offset_first,
                                         offset_first + 1,
                                         handle.get_stream());
    }
    if (tmp_storage_bytes > d_tmp_storage.size()) {
      d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
    }
    if (weights) {
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          indices + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          (*weights) + h_edge_offsets[i],
                                          (*segment_sorted_weights).data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
    } else {
      cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                         tmp_storage_bytes,
                                         indices + h_edge_offsets[i],
                                         segment_sorted_indices.data(),
                                         h_edge_offsets[i + 1] - h_edge_offsets[i],
                                         h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                         offset_first,
                                         offset_first + 1,
                                         handle.get_stream());
    }
    thrust::copy(handle.get_thrust_policy(),
                 segment_sorted_indices.begin(),
                 segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                 indices + h_edge_offsets[i]);
    if (weights) {
      thrust::copy(handle.get_thrust_policy(),
                   (*segment_sorted_weights).begin(),
                   (*segment_sorted_weights).begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   (*weights) + h_edge_offsets[i]);
    }
  }
}

}  // namespace detail

}  // namespace cugraph

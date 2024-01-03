/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <prims/kv_store.cuh>

#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cuda/functional>

#include <optional>

namespace cugraph {

namespace {

template <typename vertex_t, typename weight_t, typename edge_id_t, typename edge_type_t>
struct edge_order_t {
  thrust::optional<raft::device_span<size_t const>> edgelist_label_offsets{thrust::nullopt};
  thrust::optional<raft::device_span<int32_t const>> edgelist_hops{thrust::nullopt};
  raft::device_span<vertex_t const> edgelist_majors{};
  raft::device_span<vertex_t const> edgelist_minors{};

  __device__ bool operator()(size_t l_idx, size_t r_idx) const
  {
    if (edgelist_label_offsets) {
      auto l_label = thrust::distance((*edgelist_label_offsets).begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          (*edgelist_label_offsets).begin() + 1,
                                                          (*edgelist_label_offsets).end(),
                                                          (*edgelist_label_offsets)[0] + l_idx));
      auto r_label = thrust::distance((*edgelist_label_offsets).begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          (*edgelist_label_offsets).begin() + 1,
                                                          (*edgelist_label_offsets).end(),
                                                          (*edgelist_label_offsets)[0] + r_idx));
      if (l_label != r_label) { return l_label < r_label; }
    }

    if (edgelist_hops) {
      auto l_hop = (*edgelist_hops)[l_idx];
      auto r_hop = (*edgelist_hops)[r_idx];
      if (l_hop != r_hop) { return l_hop < r_hop; }
    }

    auto l_major = edgelist_majors[l_idx];
    auto r_major = edgelist_majors[r_idx];
    if (l_major != r_major) { return l_major < r_major; }

    auto l_minor = edgelist_minors[l_idx];
    auto r_minor = edgelist_minors[r_idx];
    if (l_minor != r_minor) { return l_minor < r_minor; }

    return l_idx < r_idx;
  }
};

template <typename vertex_t>
struct is_first_in_run_t {
  thrust::optional<raft::device_span<size_t const>> edgelist_label_offsets{thrust::nullopt};
  thrust::optional<raft::device_span<int32_t const>> edgelist_hops{thrust::nullopt};
  raft::device_span<vertex_t const> edgelist_majors{};

  __device__ bool operator()(size_t i) const
  {
    if (i == 0) return true;
    if (edgelist_label_offsets) {
      auto prev_label = thrust::distance((*edgelist_label_offsets).begin() + 1,
                                         thrust::upper_bound(thrust::seq,
                                                             (*edgelist_label_offsets).begin() + 1,
                                                             (*edgelist_label_offsets).end(),
                                                             i - 1));
      auto this_label = thrust::distance(
        (*edgelist_label_offsets).begin() + 1,
        thrust::upper_bound(
          thrust::seq, (*edgelist_label_offsets).begin() + 1, (*edgelist_label_offsets).end(), i));
      if (this_label != prev_label) { return true; }
    }
    if (edgelist_hops) {
      auto prev_hop = (*edgelist_hops)[i - 1];
      auto this_hop = (*edgelist_hops)[i];
      if (this_hop != prev_hop) { return true; }
    }
    return edgelist_majors[i] != edgelist_majors[i - 1];
  }
};

template <typename label_index_t>
struct compute_label_index_t {
  raft::device_span<size_t const> edgelist_label_offsets{};

  __device__ label_index_t operator()(size_t i) const
  {
    return static_cast<label_index_t>(thrust::distance(
      edgelist_label_offsets.begin() + 1,
      thrust::upper_bound(
        thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), i)));
  }
};

template <typename label_index_t>
struct optionally_compute_label_index_t {
  thrust::optional<raft::device_span<size_t const>> edgelist_label_offsets{thrust::nullopt};

  __device__ label_index_t operator()(size_t i) const
  {
    return edgelist_label_offsets ? static_cast<label_index_t>(thrust::distance(
                                      (*edgelist_label_offsets).begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          (*edgelist_label_offsets).begin() + 1,
                                                          (*edgelist_label_offsets).end(),
                                                          i)))
                                  : label_index_t{0};
  }
};

template <typename label_index_t,
          typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
void check_input_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t> const& edgelist_srcs,
  rmm::device_uvector<vertex_t> const& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>> const& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>> const& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>> const& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>> const& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!edgelist_label_offsets || (std::get<1>(*edgelist_label_offsets) <=
                                              std::numeric_limits<label_index_t>::max()),
                  "Invalid input arguments: current implementation assumes that the number of "
                  "unique labels is no larger than std::numeric_limits<uint32_t>::max().");

  CUGRAPH_EXPECTS(
    !edgelist_label_offsets.has_value() ||
      (std::get<0>(*edgelist_label_offsets).size() == std::get<1>(*edgelist_label_offsets) + 1),
    "Invalid input arguments: if edgelist_label_offsets is valid, "
    "std::get<0>(*edgelist_label_offsets).size() (size of the offset array) should be "
    "std::get<1>(*edgelist_label_offsets) (number of unique labels) + 1.");

  CUGRAPH_EXPECTS(
    !edgelist_hops || (std::get<1>(*edgelist_hops) <= std::numeric_limits<int32_t>::max()),
    "Invalid input arguments: current implementation assumes that the number of "
    "hops is no larger than std::numeric_limits<int32_t>::max().");
  CUGRAPH_EXPECTS(!edgelist_hops || std::get<1>(*edgelist_hops) > 0,
                  "Invlaid input arguments: number of hops should be larger than 0 if "
                  "edgelist_hops.has_value() is true.");

  CUGRAPH_EXPECTS(
    edgelist_srcs.size() == edgelist_dsts.size(),
    "Invalid input arguments: edgelist_srcs.size() and edgelist_dsts.size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_weights.has_value() || (edgelist_srcs.size() == (*edgelist_weights).size()),
    "Invalid input arguments: if edgelist_weights is valid, std::get<0>(*edgelist_weights).size() "
    "and edgelist_srcs.size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_ids.has_value() || (edgelist_srcs.size() == (*edgelist_edge_ids).size()),
    "Invalid input arguments: if edgelist_edge_ids is valid, "
    "std::get<0>(*edgelist_edge_ids).size() and edgelist_srcs.size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_types.has_value() || (edgelist_srcs.size() == (*edgelist_edge_types).size()),
    "Invalid input arguments: if edgelist_edge_types is valid, "
    "std::get<0>(*edgelist_edge_types).size() and edgelist_srcs.size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_hops.has_value() || (edgelist_srcs.size() == std::get<0>(*edgelist_hops).size()),
    "Invalid input arguments: if edgelist_hops is valid, std::get<0>(*edgelist_hops).size() and "
    "edgelist_srcs.size() should coincide.");

  if (do_expensive_check) {
    if (edgelist_label_offsets) {
      CUGRAPH_EXPECTS(thrust::is_sorted(handle.get_thrust_policy(),
                                        std::get<0>(*edgelist_label_offsets).begin(),
                                        std::get<0>(*edgelist_label_offsets).end()),
                      "Invalid input arguments: if edgelist_label_offsets is valid, "
                      "std::get<0>(*edgelist_label_offsets) should be sorted.");
      size_t back_element{};
      raft::update_host(
        &back_element,
        std::get<0>(*edgelist_label_offsets).data() + std::get<1>(*edgelist_label_offsets),
        size_t{1},
        handle.get_stream());
      handle.get_stream();
      CUGRAPH_EXPECTS(
        back_element == edgelist_srcs.size(),
        "Invalid input arguments: if edgelist_label_offsets is valid, the last element of "
        "std::get<0>(*edgelist_label_offsets) and edgelist_srcs.size() should coincide.");
    }
  }
}

// output sorted by (primary key:label_index, secondary key:vertex)
template <typename vertex_t, typename label_index_t>
std::tuple<std::optional<rmm::device_uvector<label_index_t>> /* label indices */,
           rmm::device_uvector<vertex_t> /* vertices */,
           std::optional<rmm::device_uvector<int32_t>> /* minimum hops for the vertices */,
           std::optional<rmm::device_uvector<size_t>> /* label offsets for the output */>
compute_min_hop_for_unique_label_vertex_pairs(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  std::optional<raft::device_span<int32_t const>> hops,
  std::optional<raft::device_span<label_index_t const>> label_indices,
  std::optional<raft::device_span<size_t const>> label_offsets)
{
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    (1 << 20) /* tuning parameter */;  // for segmented sort

  if (label_indices) {
    auto num_labels = (*label_offsets).size() - 1;

    rmm::device_uvector<label_index_t> tmp_label_indices((*label_indices).size(),
                                                         handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*label_indices).begin(),
                 (*label_indices).end(),
                 tmp_label_indices.begin());

    rmm::device_uvector<vertex_t> tmp_vertices(0, handle.get_stream());
    std::optional<rmm::device_uvector<int32_t>> tmp_hops{std::nullopt};

    if (hops) {
      tmp_vertices.resize(vertices.size(), handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(), vertices.begin(), vertices.end(), tmp_vertices.begin());
      tmp_hops = rmm::device_uvector<int32_t>((*hops).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(), (*hops).begin(), (*hops).end(), (*tmp_hops).begin());

      auto triplet_first = thrust::make_zip_iterator(
        tmp_label_indices.begin(), tmp_vertices.begin(), (*tmp_hops).begin());
      thrust::sort(
        handle.get_thrust_policy(), triplet_first, triplet_first + tmp_label_indices.size());
      auto key_first   = thrust::make_zip_iterator(tmp_label_indices.begin(), tmp_vertices.begin());
      auto num_uniques = static_cast<size_t>(
        thrust::distance(key_first,
                         thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                              key_first,
                                                              key_first + tmp_label_indices.size(),
                                                              (*tmp_hops).begin()))));
      tmp_label_indices.resize(num_uniques, handle.get_stream());
      tmp_vertices.resize(num_uniques, handle.get_stream());
      (*tmp_hops).resize(num_uniques, handle.get_stream());
      tmp_label_indices.shrink_to_fit(handle.get_stream());
      tmp_vertices.shrink_to_fit(handle.get_stream());
      (*tmp_hops).shrink_to_fit(handle.get_stream());
    } else {
      rmm::device_uvector<vertex_t> segment_sorted_vertices(vertices.size(), handle.get_stream());

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

      auto [h_label_offsets, h_edge_offsets] =
        detail::compute_offset_aligned_edge_chunks(handle,
                                                   (*label_offsets).data(),
                                                   num_labels,
                                                   vertices.size(),
                                                   approx_edges_to_sort_per_iteration);
      auto num_chunks = h_label_offsets.size() - 1;

      for (size_t i = 0; i < num_chunks; ++i) {
        size_t tmp_storage_bytes{0};

        auto offset_first =
          thrust::make_transform_iterator((*label_offsets).data() + h_label_offsets[i],
                                          detail::shift_left_t<size_t>{h_edge_offsets[i]});
        cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                           tmp_storage_bytes,
                                           vertices.begin() + h_edge_offsets[i],
                                           segment_sorted_vertices.begin() + h_edge_offsets[i],
                                           h_edge_offsets[i + 1] - h_edge_offsets[i],
                                           h_label_offsets[i + 1] - h_label_offsets[i],
                                           offset_first,
                                           offset_first + 1,
                                           handle.get_stream());

        if (tmp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
        }

        cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                           tmp_storage_bytes,
                                           vertices.begin() + h_edge_offsets[i],
                                           segment_sorted_vertices.begin() + h_edge_offsets[i],
                                           h_edge_offsets[i + 1] - h_edge_offsets[i],
                                           h_label_offsets[i + 1] - h_label_offsets[i],
                                           offset_first,
                                           offset_first + 1,
                                           handle.get_stream());
      }
      d_tmp_storage.resize(0, handle.get_stream());
      d_tmp_storage.shrink_to_fit(handle.get_stream());

      auto pair_first =
        thrust::make_zip_iterator(tmp_label_indices.begin(), segment_sorted_vertices.begin());
      auto num_uniques = static_cast<size_t>(thrust::distance(
        pair_first,
        thrust::unique(
          handle.get_thrust_policy(), pair_first, pair_first + tmp_label_indices.size())));
      tmp_label_indices.resize(num_uniques, handle.get_stream());
      segment_sorted_vertices.resize(num_uniques, handle.get_stream());
      tmp_label_indices.shrink_to_fit(handle.get_stream());
      segment_sorted_vertices.shrink_to_fit(handle.get_stream());

      tmp_vertices = std::move(segment_sorted_vertices);
    }

    rmm::device_uvector<size_t> tmp_label_offsets(num_labels + 1, handle.get_stream());
    tmp_label_offsets.set_element_to_zero_async(0, handle.get_stream());
    thrust::upper_bound(handle.get_thrust_policy(),
                        tmp_label_indices.begin(),
                        tmp_label_indices.end(),
                        thrust::make_counting_iterator(size_t{0}),
                        thrust::make_counting_iterator(num_labels),
                        tmp_label_offsets.begin() + 1);

    return std::make_tuple(std::move(tmp_label_indices),
                           std::move(tmp_vertices),
                           std::move(tmp_hops),
                           std::move(tmp_label_offsets));
  } else {
    rmm::device_uvector<vertex_t> tmp_vertices(vertices.size(), handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), vertices.begin(), vertices.end(), tmp_vertices.begin());

    if (hops) {
      rmm::device_uvector<int32_t> tmp_hops((*hops).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(), (*hops).begin(), (*hops).end(), tmp_hops.begin());

      auto pair_first = thrust::make_zip_iterator(
        tmp_vertices.begin(), tmp_hops.begin());  // vertex is a primary key, hop is a secondary key
      thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + tmp_vertices.size());
      tmp_vertices.resize(
        thrust::distance(tmp_vertices.begin(),
                         thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                              tmp_vertices.begin(),
                                                              tmp_vertices.end(),
                                                              tmp_hops.begin()))),
        handle.get_stream());
      tmp_hops.resize(tmp_vertices.size(), handle.get_stream());

      return std::make_tuple(
        std::nullopt, std::move(tmp_vertices), std::move(tmp_hops), std::nullopt);
    } else {
      thrust::sort(handle.get_thrust_policy(), tmp_vertices.begin(), tmp_vertices.end());
      tmp_vertices.resize(
        thrust::distance(
          tmp_vertices.begin(),
          thrust::unique(handle.get_thrust_policy(), tmp_vertices.begin(), tmp_vertices.end())),
        handle.get_stream());
      tmp_vertices.shrink_to_fit(handle.get_stream());

      return std::make_tuple(std::nullopt, std::move(tmp_vertices), std::nullopt, std::nullopt);
    }
  }
}

template <typename vertex_t, typename label_index_t>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<label_index_t>>>
compute_renumber_map(raft::handle_t const& handle,
                     raft::device_span<vertex_t const> edgelist_majors,
                     raft::device_span<vertex_t const> edgelist_minors,
                     std::optional<raft::device_span<int32_t const>> edgelist_hops,
                     std::optional<raft::device_span<size_t const>> edgelist_label_offsets)
{
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    (1 << 20) /* tuning parameter */;  // for segmented sort

  std::optional<rmm::device_uvector<label_index_t>> edgelist_label_indices{std::nullopt};
  if (edgelist_label_offsets) {
    edgelist_label_indices =
      detail::expand_sparse_offsets(*edgelist_label_offsets, label_index_t{0}, handle.get_stream());
  }

  auto [unique_label_major_pair_label_indices,
        unique_label_major_pair_vertices,
        unique_label_major_pair_hops,
        unique_label_major_pair_label_offsets] =
    compute_min_hop_for_unique_label_vertex_pairs(
      handle,
      edgelist_majors,
      edgelist_hops,
      edgelist_label_indices ? std::make_optional<raft::device_span<label_index_t const>>(
                                 (*edgelist_label_indices).data(), (*edgelist_label_indices).size())
                             : std::nullopt,
      edgelist_label_offsets);

  auto [unique_label_minor_pair_label_indices,
        unique_label_minor_pair_vertices,
        unique_label_minor_pair_hops,
        unique_label_minor_pair_label_offsets] =
    compute_min_hop_for_unique_label_vertex_pairs(
      handle,
      edgelist_minors,
      edgelist_hops,
      edgelist_label_indices ? std::make_optional<raft::device_span<label_index_t const>>(
                                 (*edgelist_label_indices).data(), (*edgelist_label_indices).size())
                             : std::nullopt,
      edgelist_label_offsets);

  edgelist_label_indices = std::nullopt;

  if (edgelist_label_offsets) {
    auto num_labels = (*edgelist_label_offsets).size() - 1;

    rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
    rmm::device_uvector<label_index_t> renumber_map_label_indices(0, handle.get_stream());

    renumber_map.reserve((*unique_label_major_pair_label_indices).size() +
                           (*unique_label_minor_pair_label_indices).size(),
                         handle.get_stream());
    renumber_map_label_indices.reserve(renumber_map.capacity(), handle.get_stream());

    auto num_chunks = (edgelist_majors.size() + (approx_edges_to_sort_per_iteration - 1)) /
                      approx_edges_to_sort_per_iteration;
    auto chunk_size = (num_chunks > 0) ? ((num_labels + (num_chunks - 1)) / num_chunks) : 0;

    size_t copy_offset{0};
    for (size_t i = 0; i < num_chunks; ++i) {
      auto major_start_offset =
        (*unique_label_major_pair_label_offsets).element(chunk_size * i, handle.get_stream());
      auto major_end_offset =
        (*unique_label_major_pair_label_offsets)
          .element(std::min(chunk_size * (i + 1), num_labels), handle.get_stream());
      auto minor_start_offset =
        (*unique_label_minor_pair_label_offsets).element(chunk_size * i, handle.get_stream());
      auto minor_end_offset =
        (*unique_label_minor_pair_label_offsets)
          .element(std::min(chunk_size * (i + 1), num_labels), handle.get_stream());

      rmm::device_uvector<label_index_t> merged_label_indices(
        (major_end_offset - major_start_offset) + (minor_end_offset - minor_start_offset),
        handle.get_stream());
      rmm::device_uvector<vertex_t> merged_vertices(merged_label_indices.size(),
                                                    handle.get_stream());
      rmm::device_uvector<int8_t> merged_flags(merged_label_indices.size(), handle.get_stream());

      if (edgelist_hops) {
        rmm::device_uvector<int32_t> merged_hops(merged_label_indices.size(), handle.get_stream());
        auto major_quad_first =
          thrust::make_zip_iterator((*unique_label_major_pair_label_indices).begin(),
                                    unique_label_major_pair_vertices.begin(),
                                    (*unique_label_major_pair_hops).begin(),
                                    thrust::make_constant_iterator(int8_t{0}));
        auto minor_quad_first =
          thrust::make_zip_iterator((*unique_label_minor_pair_label_indices).begin(),
                                    unique_label_minor_pair_vertices.begin(),
                                    (*unique_label_minor_pair_hops).begin(),
                                    thrust::make_constant_iterator(int8_t{1}));
        thrust::merge(handle.get_thrust_policy(),
                      major_quad_first + major_start_offset,
                      major_quad_first + major_end_offset,
                      minor_quad_first + minor_start_offset,
                      minor_quad_first + minor_end_offset,
                      thrust::make_zip_iterator(merged_label_indices.begin(),
                                                merged_vertices.begin(),
                                                merged_hops.begin(),
                                                merged_flags.begin()));

        auto unique_key_first =
          thrust::make_zip_iterator(merged_label_indices.begin(), merged_vertices.begin());
        merged_label_indices.resize(
          thrust::distance(
            unique_key_first,
            thrust::get<0>(thrust::unique_by_key(
              handle.get_thrust_policy(),
              unique_key_first,
              unique_key_first + merged_label_indices.size(),
              thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin())))),
          handle.get_stream());
        merged_vertices.resize(merged_label_indices.size(), handle.get_stream());
        merged_hops.resize(merged_label_indices.size(), handle.get_stream());
        merged_flags.resize(merged_label_indices.size(), handle.get_stream());
        auto sort_key_first = thrust::make_zip_iterator(
          merged_label_indices.begin(), merged_hops.begin(), merged_flags.begin());
        thrust::sort_by_key(handle.get_thrust_policy(),
                            sort_key_first,
                            sort_key_first + merged_label_indices.size(),
                            merged_vertices.begin());
      } else {
        auto major_triplet_first =
          thrust::make_zip_iterator((*unique_label_major_pair_label_indices).begin(),
                                    unique_label_major_pair_vertices.begin(),
                                    thrust::make_constant_iterator(int8_t{0}));
        auto minor_triplet_first =
          thrust::make_zip_iterator((*unique_label_minor_pair_label_indices).begin(),
                                    unique_label_minor_pair_vertices.begin(),
                                    thrust::make_constant_iterator(int8_t{1}));
        thrust::merge(
          handle.get_thrust_policy(),
          major_triplet_first + major_start_offset,
          major_triplet_first + major_end_offset,
          minor_triplet_first + minor_start_offset,
          minor_triplet_first + minor_end_offset,
          thrust::make_zip_iterator(
            merged_label_indices.begin(), merged_vertices.begin(), merged_flags.begin()));

        auto unique_key_first =
          thrust::make_zip_iterator(merged_label_indices.begin(), merged_vertices.begin());
        merged_label_indices.resize(
          thrust::distance(
            unique_key_first,
            thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                 unique_key_first,
                                                 unique_key_first + merged_label_indices.size(),
                                                 merged_flags.begin()))),
          handle.get_stream());
        merged_vertices.resize(merged_label_indices.size(), handle.get_stream());
        merged_flags.resize(merged_label_indices.size(), handle.get_stream());
        auto sort_key_first =
          thrust::make_zip_iterator(merged_label_indices.begin(), merged_flags.begin());
        thrust::sort_by_key(handle.get_thrust_policy(),
                            sort_key_first,
                            sort_key_first + merged_label_indices.size(),
                            merged_vertices.begin());
      }

      renumber_map.resize(copy_offset + merged_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   merged_vertices.begin(),
                   merged_vertices.end(),
                   renumber_map.begin() + copy_offset);
      renumber_map_label_indices.resize(copy_offset + merged_label_indices.size(),
                                        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   merged_label_indices.begin(),
                   merged_label_indices.end(),
                   renumber_map_label_indices.begin() + copy_offset);

      copy_offset += merged_vertices.size();
    }

    renumber_map.shrink_to_fit(handle.get_stream());
    renumber_map_label_indices.shrink_to_fit(handle.get_stream());

    return std::make_tuple(std::move(renumber_map), std::move(renumber_map_label_indices));
  } else {
    if (edgelist_hops) {
      rmm::device_uvector<vertex_t> merged_vertices(
        unique_label_major_pair_vertices.size() + unique_label_minor_pair_vertices.size(),
        handle.get_stream());
      rmm::device_uvector<int32_t> merged_hops(merged_vertices.size(), handle.get_stream());
      rmm::device_uvector<int8_t> merged_flags(merged_vertices.size(), handle.get_stream());
      auto major_triplet_first =
        thrust::make_zip_iterator(unique_label_major_pair_vertices.begin(),
                                  (*unique_label_major_pair_hops).begin(),
                                  thrust::make_constant_iterator(int8_t{0}));
      auto minor_triplet_first =
        thrust::make_zip_iterator(unique_label_minor_pair_vertices.begin(),
                                  (*unique_label_minor_pair_hops).begin(),
                                  thrust::make_constant_iterator(int8_t{1}));
      thrust::merge(handle.get_thrust_policy(),
                    major_triplet_first,
                    major_triplet_first + unique_label_major_pair_vertices.size(),
                    minor_triplet_first,
                    minor_triplet_first + unique_label_minor_pair_vertices.size(),
                    thrust::make_zip_iterator(
                      merged_vertices.begin(), merged_hops.begin(), merged_flags.begin()));

      unique_label_major_pair_vertices.resize(0, handle.get_stream());
      unique_label_major_pair_vertices.shrink_to_fit(handle.get_stream());
      unique_label_major_pair_hops = std::nullopt;
      unique_label_minor_pair_vertices.resize(0, handle.get_stream());
      unique_label_minor_pair_vertices.shrink_to_fit(handle.get_stream());
      unique_label_minor_pair_hops = std::nullopt;

      merged_vertices.resize(
        thrust::distance(merged_vertices.begin(),
                         thrust::get<0>(thrust::unique_by_key(
                           handle.get_thrust_policy(),
                           merged_vertices.begin(),
                           merged_vertices.end(),
                           thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin())))),
        handle.get_stream());
      merged_hops.resize(merged_vertices.size(), handle.get_stream());
      merged_flags.resize(merged_vertices.size(), handle.get_stream());

      auto sort_key_first = thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          sort_key_first,
                          sort_key_first + merged_hops.size(),
                          merged_vertices.begin());

      return std::make_tuple(std::move(merged_vertices), std::nullopt);
    } else {
      rmm::device_uvector<vertex_t> output_vertices(unique_label_minor_pair_vertices.size(),
                                                    handle.get_stream());
      auto output_last = thrust::set_difference(handle.get_thrust_policy(),
                                                unique_label_minor_pair_vertices.begin(),
                                                unique_label_minor_pair_vertices.end(),
                                                unique_label_major_pair_vertices.begin(),
                                                unique_label_major_pair_vertices.end(),
                                                output_vertices.begin());

      auto num_unique_majors = unique_label_major_pair_vertices.size();
      auto renumber_map      = std::move(unique_label_major_pair_vertices);
      renumber_map.resize(
        renumber_map.size() + thrust::distance(output_vertices.begin(), output_last),
        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   output_vertices.begin(),
                   output_last,
                   renumber_map.begin() + num_unique_majors);

      return std::make_tuple(std::move(renumber_map), std::nullopt);
    }
  }
}

// this function does not reorder edges (the i'th returned edge is the renumbered output of the i'th
// input edge)
template <typename vertex_t, typename label_index_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<size_t>>>
renumber_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_majors,
  rmm::device_uvector<vertex_t>&& edgelist_minors,
  std::optional<std::tuple<raft::device_span<int32_t const>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool do_expensive_check)
{
  // 1. compute renumber_map

  auto [renumber_map, renumber_map_label_indices] = compute_renumber_map<vertex_t, label_index_t>(
    handle,
    raft::device_span<vertex_t const>(edgelist_majors.data(), edgelist_majors.size()),
    raft::device_span<vertex_t const>(edgelist_minors.data(), edgelist_minors.size()),
    edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                      std::get<0>(*edgelist_hops).data(), std::get<0>(*edgelist_hops).size())
                  : std::nullopt,
    edgelist_label_offsets
      ? std::make_optional<raft::device_span<size_t const>>(std::get<0>(*edgelist_label_offsets))
      : std::nullopt);

  // 2. compute renumber map offsets for each label

  std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{};
  if (edgelist_label_offsets) {
    auto num_unique_labels = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator((*renumber_map_label_indices).size()),
      detail::is_first_in_run_t<label_index_t const*>{(*renumber_map_label_indices).data()});
    rmm::device_uvector<label_index_t> unique_label_indices(num_unique_labels, handle.get_stream());
    rmm::device_uvector<vertex_t> vertex_counts(num_unique_labels, handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          (*renumber_map_label_indices).begin(),
                          (*renumber_map_label_indices).end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_label_indices.begin(),
                          vertex_counts.begin());

    renumber_map_label_offsets =
      rmm::device_uvector<size_t>(std::get<1>(*edgelist_label_offsets) + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 (*renumber_map_label_offsets).begin(),
                 (*renumber_map_label_offsets).end(),
                 size_t{0});
    thrust::scatter(handle.get_thrust_policy(),
                    vertex_counts.begin(),
                    vertex_counts.end(),
                    unique_label_indices.begin(),
                    (*renumber_map_label_offsets).begin() + 1);

    thrust::inclusive_scan(handle.get_thrust_policy(),
                           (*renumber_map_label_offsets).begin(),
                           (*renumber_map_label_offsets).end(),
                           (*renumber_map_label_offsets).begin());
  }

  // 3. renumber input edges

  if (edgelist_label_offsets) {
    rmm::device_uvector<vertex_t> new_vertices(renumber_map.size(), handle.get_stream());
    thrust::tabulate(handle.get_thrust_policy(),
                     new_vertices.begin(),
                     new_vertices.end(),
                     [label_indices = raft::device_span<label_index_t const>(
                        (*renumber_map_label_indices).data(), (*renumber_map_label_indices).size()),
                      renumber_map_label_offsets = raft::device_span<size_t const>(
                        (*renumber_map_label_offsets).data(),
                        (*renumber_map_label_offsets).size())] __device__(size_t i) {
                       auto label_index        = label_indices[i];
                       auto label_start_offset = renumber_map_label_offsets[label_index];
                       return static_cast<vertex_t>(i - label_start_offset);
                     });

    (*renumber_map_label_indices).resize(0, handle.get_stream());
    (*renumber_map_label_indices).shrink_to_fit(handle.get_stream());

    auto num_labels = std::get<0>(*edgelist_label_offsets).size();

    rmm::device_uvector<vertex_t> segment_sorted_renumber_map(renumber_map.size(),
                                                              handle.get_stream());
    rmm::device_uvector<vertex_t> segment_sorted_new_vertices(new_vertices.size(),
                                                              handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    auto approx_edges_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20) /* tuning parameter */;  // for segmented sort

    auto [h_label_offsets, h_edge_offsets] = detail::compute_offset_aligned_edge_chunks(
      handle,
      (*renumber_map_label_offsets).data(),
      static_cast<size_t>((*renumber_map_label_offsets).size() - 1),
      renumber_map.size(),
      approx_edges_to_sort_per_iteration);
    auto num_chunks = h_label_offsets.size() - 1;

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};

      auto offset_first =
        thrust::make_transform_iterator((*renumber_map_label_offsets).data() + h_label_offsets[i],
                                        detail::shift_left_t<size_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          renumber_map.begin() + h_edge_offsets[i],
                                          segment_sorted_renumber_map.begin() + h_edge_offsets[i],
                                          new_vertices.begin() + h_edge_offsets[i],
                                          segment_sorted_new_vertices.begin() + h_edge_offsets[i],
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_label_offsets[i + 1] - h_label_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());

      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }

      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          renumber_map.begin() + h_edge_offsets[i],
                                          segment_sorted_renumber_map.begin() + h_edge_offsets[i],
                                          new_vertices.begin() + h_edge_offsets[i],
                                          segment_sorted_new_vertices.begin() + h_edge_offsets[i],
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_label_offsets[i + 1] - h_label_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
    }
    new_vertices.resize(0, handle.get_stream());
    d_tmp_storage.resize(0, handle.get_stream());
    new_vertices.shrink_to_fit(handle.get_stream());
    d_tmp_storage.shrink_to_fit(handle.get_stream());

    auto edgelist_label_indices = detail::expand_sparse_offsets(
      std::get<0>(*edgelist_label_offsets), label_index_t{0}, handle.get_stream());

    auto pair_first =
      thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_label_indices.begin());
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_majors.size(),
      edgelist_majors.begin(),
      [renumber_map_label_offsets = raft::device_span<size_t const>(
         (*renumber_map_label_offsets).data(), (*renumber_map_label_offsets).size()),
       old_vertices = raft::device_span<vertex_t const>(segment_sorted_renumber_map.data(),
                                                        segment_sorted_renumber_map.size()),
       new_vertices = raft::device_span<vertex_t const>(
         segment_sorted_new_vertices.data(),
         segment_sorted_new_vertices.size())] __device__(auto pair) {
        auto old_vertex         = thrust::get<0>(pair);
        auto label_index        = thrust::get<1>(pair);
        auto label_start_offset = renumber_map_label_offsets[label_index];
        auto label_end_offset   = renumber_map_label_offsets[label_index + 1];
        auto it                 = thrust::lower_bound(thrust::seq,
                                      old_vertices.begin() + label_start_offset,
                                      old_vertices.begin() + label_end_offset,
                                      old_vertex);
        assert(*it == old_vertex);
        return *(new_vertices.begin() + thrust::distance(old_vertices.begin(), it));
      });

    pair_first = thrust::make_zip_iterator(edgelist_minors.begin(), edgelist_label_indices.begin());
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_minors.size(),
      edgelist_minors.begin(),
      [renumber_map_label_offsets = raft::device_span<size_t const>(
         (*renumber_map_label_offsets).data(), (*renumber_map_label_offsets).size()),
       old_vertices = raft::device_span<vertex_t const>(segment_sorted_renumber_map.data(),
                                                        segment_sorted_renumber_map.size()),
       new_vertices = raft::device_span<vertex_t const>(
         segment_sorted_new_vertices.data(),
         segment_sorted_new_vertices.size())] __device__(auto pair) {
        auto old_vertex         = thrust::get<0>(pair);
        auto label_index        = thrust::get<1>(pair);
        auto label_start_offset = renumber_map_label_offsets[label_index];
        auto label_end_offset   = renumber_map_label_offsets[label_index + 1];
        auto it                 = thrust::lower_bound(thrust::seq,
                                      old_vertices.begin() + label_start_offset,
                                      old_vertices.begin() + label_end_offset,
                                      old_vertex);
        assert(*it == old_vertex);
        return new_vertices[thrust::distance(old_vertices.begin(), it)];
      });
  } else {
    kv_store_t<vertex_t, vertex_t, false> kv_store(renumber_map.begin(),
                                                   renumber_map.end(),
                                                   thrust::make_counting_iterator(vertex_t{0}),
                                                   std::numeric_limits<vertex_t>::max(),
                                                   std::numeric_limits<vertex_t>::max(),
                                                   handle.get_stream());
    auto kv_store_view = kv_store.view();

    kv_store_view.find(
      edgelist_majors.begin(), edgelist_majors.end(), edgelist_majors.begin(), handle.get_stream());
    kv_store_view.find(
      edgelist_minors.begin(), edgelist_minors.end(), edgelist_minors.begin(), handle.get_stream());
  }

  return std::make_tuple(std::move(edgelist_majors),
                         std::move(edgelist_minors),
                         std::move(renumber_map),
                         std::move(renumber_map_label_offsets));
}

template <typename IndexIterator, typename ValueIterator>
void permute_array(raft::handle_t const& handle,
                   IndexIterator index_first,
                   IndexIterator index_last,
                   ValueIterator value_first /* [INOUT] */)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  auto tmp_buffer = allocate_dataframe_buffer<value_t>(thrust::distance(index_first, index_last),
                                                       handle.get_stream());
  thrust::gather(handle.get_thrust_policy(),
                 index_first,
                 index_last,
                 value_first,
                 get_dataframe_buffer_begin(tmp_buffer));
  thrust::copy(handle.get_thrust_policy(),
               get_dataframe_buffer_begin(tmp_buffer),
               get_dataframe_buffer_end(tmp_buffer),
               value_first);
}

// key: ((label), (hop), major, minor)
template <typename vertex_t, typename weight_t, typename edge_id_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_id_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>>
sort_sampled_edge_tuples(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_majors,
  rmm::device_uvector<vertex_t>&& edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets)
{
  std::vector<size_t> h_label_offsets{};
  std::vector<size_t> h_edge_offsets{};

  if (edgelist_label_offsets) {
    auto approx_edges_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20) /* tuning parameter */;  // for sorts in chunks

    std::tie(h_label_offsets, h_edge_offsets) =
      detail::compute_offset_aligned_edge_chunks(handle,
                                                 std::get<0>(*edgelist_label_offsets).data(),
                                                 std::get<1>(*edgelist_label_offsets),
                                                 edgelist_majors.size(),
                                                 approx_edges_to_sort_per_iteration);
  } else {
    h_label_offsets = {0, 1};
    h_edge_offsets  = {0, edgelist_majors.size()};
  }

  auto num_chunks = h_label_offsets.size() - 1;
  for (size_t i = 0; i < num_chunks; ++i) {
    rmm::device_uvector<size_t> indices(h_edge_offsets[i + 1] - h_edge_offsets[i],
                                        handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
    edge_order_t<vertex_t, weight_t, edge_id_t, edge_type_t> edge_order_comp{
      edgelist_label_offsets ? thrust::make_optional<raft::device_span<size_t const>>(
                                 std::get<0>(*edgelist_label_offsets).data() + h_label_offsets[i],
                                 (h_label_offsets[i + 1] - h_label_offsets[i]) + 1)
                             : thrust::nullopt,
      edgelist_hops ? thrust::make_optional<raft::device_span<int32_t const>>(
                        std::get<0>(*edgelist_hops).data() + h_edge_offsets[i], indices.size())
                    : thrust::nullopt,
      raft::device_span<vertex_t const>(edgelist_majors.data() + h_edge_offsets[i], indices.size()),
      raft::device_span<vertex_t const>(edgelist_minors.data() + h_edge_offsets[i],
                                        indices.size())};
    thrust::sort(handle.get_thrust_policy(), indices.begin(), indices.end(), edge_order_comp);

    permute_array(handle,
                  indices.begin(),
                  indices.end(),
                  thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin()) +
                    h_edge_offsets[i]);

    if (edgelist_weights) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_weights).begin() + h_edge_offsets[i]);
    }

    if (edgelist_edge_ids) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_edge_ids).begin() + h_edge_offsets[i]);
    }

    if (edgelist_edge_types) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_edge_types).begin() + h_edge_offsets[i]);
    }

    if (edgelist_hops) {
      permute_array(handle,
                    indices.begin(),
                    indices.end(),
                    std::get<0>(*edgelist_hops).begin() + h_edge_offsets[i]);
    }
  }

  return std::make_tuple(std::move(edgelist_majors),
                         std::move(edgelist_minors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_hops));
}

}  // namespace

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>,     // dcsr/dcsc major vertices
           rmm::device_uvector<size_t>,                      // (d)csr/(d)csc offset values
           rmm::device_uvector<vertex_t>,                    // minor vertices
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,  // (label, hop) offsets to the (d)csr/(d)csc
                                                        // offset array
           rmm::device_uvector<vertex_t>,               // renumber map
           std::optional<rmm::device_uvector<size_t>>>  // label offsets to the renumber map
renumber_and_compress_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool src_is_major,
  bool compress_per_hop,
  bool doubly_compress,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;

  auto num_labels = edgelist_label_offsets ? std::get<1>(*edgelist_label_offsets) : size_t{1};
  auto num_hops   = edgelist_hops ? std::get<1>(*edgelist_hops) : size_t{1};

  // 1. check input arguments

  check_input_edges<label_index_t>(handle,
                                   edgelist_srcs,
                                   edgelist_dsts,
                                   edgelist_weights,
                                   edgelist_edge_ids,
                                   edgelist_edge_types,
                                   edgelist_hops,
                                   edgelist_label_offsets,
                                   do_expensive_check);

  CUGRAPH_EXPECTS(
    !doubly_compress || !compress_per_hop,
    "Invalid input arguments: compress_per_hop should be false if doubly_compress is true.");
  CUGRAPH_EXPECTS(!compress_per_hop || edgelist_hops,
                  "Invalid input arguments: edgelist_hops.has_value() should be true if "
                  "compress_per_hop is true.");

  // 2. renumber

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{std::nullopt};
  std::tie(edgelist_majors, edgelist_minors, renumber_map, renumber_map_label_offsets) =
    renumber_sampled_edgelist<vertex_t, label_index_t>(
      handle,
      std::move(edgelist_majors),
      std::move(edgelist_minors),
      edgelist_hops ? std::make_optional(std::make_tuple(
                        raft::device_span<int32_t const>(std::get<0>(*edgelist_hops).data(),
                                                         std::get<0>(*edgelist_hops).size()),
                        num_hops))
                    : std::nullopt,
      edgelist_label_offsets,
      do_expensive_check);

  // 3. sort by ((l), (h), major, minor)

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets);

  if (do_expensive_check) {
    if (!compress_per_hop && edgelist_hops) {
      rmm::device_uvector<vertex_t> min_vertices(num_labels * num_hops, handle.get_stream());
      rmm::device_uvector<vertex_t> max_vertices(min_vertices.size(), handle.get_stream());

      auto label_index_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        optionally_compute_label_index_t<label_index_t>{
          edgelist_label_offsets ? thrust::make_optional(std::get<0>(*edgelist_label_offsets))
                                 : thrust::nullopt});
      auto input_key_first =
        thrust::make_zip_iterator(label_index_first, std::get<0>(*edgelist_hops).begin());
      rmm::device_uvector<label_index_t> unique_key_label_indices(min_vertices.size(),
                                                                  handle.get_stream());
      rmm::device_uvector<int32_t> unique_key_hops(min_vertices.size(), handle.get_stream());
      auto output_key_first =
        thrust::make_zip_iterator(unique_key_label_indices.begin(), unique_key_hops.begin());

      auto output_it =
        thrust::reduce_by_key(handle.get_thrust_policy(),
                              input_key_first,
                              input_key_first + edgelist_majors.size(),
                              edgelist_majors.begin(),
                              output_key_first,
                              min_vertices.begin(),
                              thrust::equal_to<thrust::tuple<label_index_t, int32_t>>{},
                              thrust::minimum<vertex_t>{});
      auto num_unique_keys =
        static_cast<size_t>(thrust::distance(output_key_first, thrust::get<0>(output_it)));
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            edgelist_majors.begin(),
                            output_key_first,
                            max_vertices.begin(),
                            thrust::equal_to<thrust::tuple<label_index_t, int32_t>>{},
                            thrust::maximum<vertex_t>{});
      if (num_unique_keys > 1) {
        auto num_invalids = thrust::count_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{1}),
          thrust::make_counting_iterator(num_unique_keys),
          [output_key_first,
           min_vertices = raft::device_span<vertex_t const>(min_vertices.data(), num_unique_keys),
           max_vertices = raft::device_span<vertex_t const>(max_vertices.data(),
                                                            num_unique_keys)] __device__(size_t i) {
            auto prev_key = *(output_key_first + (i - 1));
            auto this_key = *(output_key_first + i);
            if (thrust::get<0>(prev_key) == thrust::get<0>(this_key)) {
              auto this_min = min_vertices[i];
              auto prev_max = max_vertices[i - 1];
              return prev_max >= this_min;
            } else {
              return false;
            }
          });
        CUGRAPH_EXPECTS(num_invalids == 0,
                        "Invalid input arguments: if @p compress_per_hop is false and @p "
                        "edgelist_hops.has_value() is true, the minimum majors with hop N + 1 "
                        "should be larger than the maximum majors with hop N after renumbering.");
      }
    }
  }

  // 4. compute offsets for ((l), (h), major) triplets with non zero neighbors (update
  // compressed_label_indices, compressed_hops, compressed_nzd_vertices, and compressed_offsets)

  auto num_uniques = thrust::count_if(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(edgelist_majors.size()),
    is_first_in_run_t<vertex_t>{
      edgelist_label_offsets ? thrust::make_optional(std::get<0>(*edgelist_label_offsets))
                             : thrust::nullopt,
      edgelist_hops ? thrust::make_optional<raft::device_span<int32_t const>>(
                        std::get<0>(*edgelist_hops).data(), std::get<0>(*edgelist_hops).size())
                    : thrust::nullopt,
      raft::device_span<vertex_t const>(
        edgelist_majors.data(),
        edgelist_majors.size())});  // number of unique ((label), (hop), major) triplets

  auto compressed_label_indices =
    edgelist_label_offsets
      ? std::make_optional<rmm::device_uvector<label_index_t>>(num_uniques, handle.get_stream())
      : std::nullopt;
  auto compressed_hops = edgelist_hops ? std::make_optional<rmm::device_uvector<int32_t>>(
                                           num_uniques, handle.get_stream())
                                       : std::nullopt;
  rmm::device_uvector<vertex_t> compressed_nzd_vertices(num_uniques, handle.get_stream());
  rmm::device_uvector<size_t> compressed_offsets(num_uniques + 1, handle.get_stream());
  compressed_offsets.set_element_to_zero_async(num_uniques, handle.get_stream());

  if (edgelist_label_offsets) {
    auto label_index_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      compute_label_index_t<label_index_t>{std::get<0>(*edgelist_label_offsets)});

    if (edgelist_hops) {
      auto input_key_first = thrust::make_zip_iterator(
        label_index_first, std::get<0>(*edgelist_hops).begin(), edgelist_majors.begin());
      auto output_key_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                        (*compressed_hops).begin(),
                                                        compressed_nzd_vertices.begin());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    } else {
      auto input_key_first  = thrust::make_zip_iterator(label_index_first, edgelist_majors.begin());
      auto output_key_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                        compressed_nzd_vertices.begin());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    }
  } else {
    if (edgelist_hops) {
      auto input_key_first =
        thrust::make_zip_iterator(std::get<0>(*edgelist_hops).begin(), edgelist_majors.begin());
      auto output_key_first =
        thrust::make_zip_iterator((*compressed_hops).begin(), compressed_nzd_vertices.begin());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    } else {
      auto input_key_first  = edgelist_majors.begin();
      auto output_key_first = compressed_nzd_vertices.begin();
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    }
  }
  thrust::exclusive_scan(handle.get_thrust_policy(),
                         compressed_offsets.begin(),
                         compressed_offsets.end(),
                         compressed_offsets.begin());

  // 5. update compressed_offsets to include zero degree vertices (if doubly_compress is false) and
  // compressed_offset_label_hop_offsets (if edgelist_label_offsets.has_value() or
  // edgelist_hops.has_value() is true)

  std::optional<rmm::device_uvector<size_t>> compressed_offset_label_hop_offsets{std::nullopt};
  if (doubly_compress) {
    if (edgelist_label_offsets || edgelist_hops) {
      rmm::device_uvector<size_t> offset_array_offsets(num_labels * num_hops + 1,
                                                       handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   offset_array_offsets.begin(),
                   offset_array_offsets.end(),
                   size_t{0});

      if (edgelist_label_offsets) {
        if (edgelist_hops) {
          auto pair_first       = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                      (*compressed_hops).begin());
          auto value_pair_first = thrust::make_transform_iterator(
            thrust::make_counting_iterator(size_t{0}),
            cuda::proclaim_return_type<thrust::tuple<label_index_t, int32_t>>(
              [num_hops] __device__(size_t i) {
                return thrust::make_tuple(static_cast<label_index_t>(i / num_hops),
                                          static_cast<int32_t>(i % num_hops));
              }));
          thrust::upper_bound(handle.get_thrust_policy(),
                              pair_first,
                              pair_first + (*compressed_label_indices).size(),
                              value_pair_first,
                              value_pair_first + (num_labels * num_hops),
                              offset_array_offsets.begin() + 1);
        } else {
          thrust::upper_bound(
            handle.get_thrust_policy(),
            (*compressed_label_indices).begin(),
            (*compressed_label_indices).end(),
            thrust::make_counting_iterator(label_index_t{0}),
            thrust::make_counting_iterator(static_cast<label_index_t>(num_labels)),
            offset_array_offsets.begin() + 1);
        }
      } else {
        thrust::upper_bound(handle.get_thrust_policy(),
                            (*compressed_hops).begin(),
                            (*compressed_hops).end(),
                            thrust::make_counting_iterator(int32_t{0}),
                            thrust::make_counting_iterator(static_cast<int32_t>(num_hops)),
                            offset_array_offsets.begin() + 1);
      }

      compressed_offset_label_hop_offsets = std::move(offset_array_offsets);
    }
  } else {  // !doubly_compress
    rmm::device_uvector<vertex_t> major_vertex_counts(num_labels * num_hops, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      major_vertex_counts.begin(),
      major_vertex_counts.end(),
      [edgelist_label_offsets = edgelist_label_offsets
                                  ? thrust::make_optional(std::get<0>(*edgelist_label_offsets))
                                  : thrust::nullopt,
       edgelist_hops          = edgelist_hops
                                  ? thrust::make_optional<raft::device_span<int32_t>>(
                             std::get<0>(*edgelist_hops).data(), std::get<0>(*edgelist_hops).size())
                                  : thrust::nullopt,
       edgelist_majors =
         raft::device_span<vertex_t const>(edgelist_majors.data(), edgelist_majors.size()),
       num_hops,
       compress_per_hop] __device__(size_t i) {
        size_t start_offset{0};
        auto end_offset         = edgelist_majors.size();
        auto label_start_offset = start_offset;
        auto label_end_offset   = end_offset;

        if (edgelist_label_offsets) {
          auto l_idx         = static_cast<label_index_t>(i / num_hops);
          start_offset       = (*edgelist_label_offsets)[l_idx];
          end_offset         = (*edgelist_label_offsets)[l_idx + 1];
          label_start_offset = start_offset;
          label_end_offset   = end_offset;
        }

        if (num_hops > 1) {
          auto h        = static_cast<int32_t>(i % num_hops);
          auto lower_it = thrust::lower_bound(thrust::seq,
                                              (*edgelist_hops).begin() + start_offset,
                                              (*edgelist_hops).begin() + end_offset,
                                              h);
          auto upper_it = thrust::upper_bound(thrust::seq,
                                              (*edgelist_hops).begin() + start_offset,
                                              (*edgelist_hops).begin() + end_offset,
                                              h);
          start_offset  = static_cast<size_t>(thrust::distance((*edgelist_hops).begin(), lower_it));
          end_offset    = static_cast<size_t>(thrust::distance((*edgelist_hops).begin(), upper_it));
        }
        if (compress_per_hop) {
          return (start_offset < end_offset) ? (edgelist_majors[end_offset - 1] + 1) : vertex_t{0};
        } else {
          if (end_offset != label_end_offset) {
            return edgelist_majors[end_offset];
          } else if (label_start_offset < label_end_offset) {
            return edgelist_majors[end_offset - 1] + 1;
          } else {
            return vertex_t{0};
          }
        }
      });

    std::optional<rmm::device_uvector<vertex_t>> minor_vertex_counts{std::nullopt};
    if (compress_per_hop) {
      minor_vertex_counts =
        rmm::device_uvector<vertex_t>(major_vertex_counts.size(), handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   (*minor_vertex_counts).begin(),
                   (*minor_vertex_counts).end(),
                   vertex_t{0});
      if (edgelist_label_offsets) {
        auto triplet_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                       (*compressed_hops).begin(),
                                                       thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(handle.get_thrust_policy(),
                         triplet_first,
                         triplet_first + compressed_nzd_vertices.size(),
                         [edgelist_minors = raft::device_span<vertex_t const>(
                            edgelist_minors.data(), edgelist_minors.size()),
                          compressed_offsets = raft::device_span<size_t const>(
                            compressed_offsets.data(), compressed_offsets.size()),
                          minor_vertex_counts = raft::device_span<vertex_t>(
                            (*minor_vertex_counts).data(), (*minor_vertex_counts).size()),
                          num_hops] __device__(auto triplet) {
                           auto nzd_v_idx    = thrust::get<2>(triplet);
                           size_t end_offset = compressed_offsets[nzd_v_idx + 1];
                           auto l_idx        = thrust::get<0>(triplet);
                           auto h            = thrust::get<1>(triplet);
                           cuda::atomic_ref<vertex_t, cuda::thread_scope_device> minor_vertex_count(
                             minor_vertex_counts[l_idx * num_hops + h]);
                           minor_vertex_count.fetch_max(edgelist_minors[end_offset - 1] + 1,
                                                        cuda::std::memory_order_relaxed);
                         });
      } else {
        auto pair_first = thrust::make_zip_iterator((*compressed_hops).begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(handle.get_thrust_policy(),
                         pair_first,
                         pair_first + compressed_nzd_vertices.size(),
                         [edgelist_minors = raft::device_span<vertex_t const>(
                            edgelist_minors.data(), edgelist_minors.size()),
                          compressed_offsets = raft::device_span<size_t const>(
                            compressed_offsets.data(), compressed_offsets.size()),
                          minor_vertex_counts = raft::device_span<vertex_t>(
                            (*minor_vertex_counts).data(), (*minor_vertex_counts).size()),
                          num_hops] __device__(auto pair) {
                           auto nzd_v_idx    = thrust::get<1>(pair);
                           size_t end_offset = compressed_offsets[nzd_v_idx + 1];
                           auto h            = thrust::get<0>(pair);
                           cuda::atomic_ref<vertex_t, cuda::thread_scope_device> minor_vertex_count(
                             minor_vertex_counts[h]);
                           minor_vertex_count.fetch_max(edgelist_minors[end_offset - 1] + 1,
                                                        cuda::std::memory_order_relaxed);
                         });
      }
    }

    rmm::device_uvector<size_t> offset_array_offsets(num_labels * num_hops + 1,
                                                     handle.get_stream());
    offset_array_offsets.set_element_to_zero_async(num_labels * num_hops, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      offset_array_offsets.begin(),
      offset_array_offsets.begin() + (num_labels * num_hops),
      [major_vertex_counts =
         raft::device_span<vertex_t const>(major_vertex_counts.data(), major_vertex_counts.size()),
       minor_vertex_counts = minor_vertex_counts
                               ? thrust::make_optional<raft::device_span<vertex_t const>>(
                                   (*minor_vertex_counts).data(), (*minor_vertex_counts).size())
                               : thrust::nullopt,
       num_hops,
       compress_per_hop] __device__(size_t i) {
        auto vertex_count = major_vertex_counts[i];
        if (num_hops > 1) {
          if (compress_per_hop) {
            for (size_t j = (i - (i % num_hops)); j < i; ++j) {
              vertex_count = cuda::std::max(vertex_count, major_vertex_counts[j]);
              vertex_count = cuda::std::max(vertex_count, (*minor_vertex_counts)[j]);
            }
          } else {
            if (i % num_hops != 0) { vertex_count -= major_vertex_counts[i - 1]; }
          }
        }
        return vertex_count;
      });
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           offset_array_offsets.begin(),
                           offset_array_offsets.end(),
                           offset_array_offsets.begin());

    auto tmp_compressed_offsets = rmm::device_uvector<size_t>(
      offset_array_offsets.back_element(handle.get_stream()) + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 tmp_compressed_offsets.begin(),
                 tmp_compressed_offsets.end(),
                 size_t{0});

    if (edgelist_label_offsets) {
      if (edgelist_hops) {
        auto triplet_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                       (*compressed_hops).begin(),
                                                       thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(
          handle.get_thrust_policy(),
          triplet_first,
          triplet_first + compressed_nzd_vertices.size(),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           offset_array_offsets = raft::device_span<size_t const>(offset_array_offsets.data(),
                                                                  offset_array_offsets.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets = raft::device_span<size_t>(tmp_compressed_offsets.data(),
                                                              tmp_compressed_offsets.size()),
           compress_per_hop,
           num_hops] __device__(auto triplet) {
            auto nzd_v_idx      = thrust::get<2>(triplet);
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            auto l_idx          = thrust::get<0>(triplet);
            auto h              = thrust::get<1>(triplet);
            tmp_compressed_offsets[offset_array_offsets[l_idx * num_hops +
                                                        (compress_per_hop ? h : int32_t{0})] +
                                   compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      } else {
        auto pair_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + compressed_nzd_vertices.size(),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           offset_array_offsets = raft::device_span<size_t const>(offset_array_offsets.data(),
                                                                  offset_array_offsets.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets = raft::device_span<size_t>(
             tmp_compressed_offsets.data(), tmp_compressed_offsets.size())] __device__(auto pair) {
            auto nzd_v_idx      = thrust::get<1>(pair);
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            auto l_idx          = thrust::get<0>(pair);
            tmp_compressed_offsets[offset_array_offsets[l_idx] +
                                   compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      }
    } else {
      if (edgelist_hops) {
        auto pair_first = thrust::make_zip_iterator((*compressed_hops).begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + compressed_nzd_vertices.size(),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           offset_array_offsets = raft::device_span<size_t const>(offset_array_offsets.data(),
                                                                  offset_array_offsets.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets = raft::device_span<size_t>(tmp_compressed_offsets.data(),
                                                              tmp_compressed_offsets.size()),
           compress_per_hop] __device__(auto pair) {
            auto nzd_v_idx      = thrust::get<1>(pair);
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            auto h              = thrust::get<0>(pair);
            tmp_compressed_offsets[offset_array_offsets[compress_per_hop ? h : int32_t{0}] +
                                   compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      } else {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(compressed_nzd_vertices.size()),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets =
             raft::device_span<size_t>(tmp_compressed_offsets.data(),
                                       tmp_compressed_offsets.size())] __device__(auto nzd_v_idx) {
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            tmp_compressed_offsets[compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      }
    }

    thrust::exclusive_scan(handle.get_thrust_policy(),
                           tmp_compressed_offsets.begin(),
                           tmp_compressed_offsets.end(),
                           tmp_compressed_offsets.begin());

    compressed_offsets = std::move(tmp_compressed_offsets);

    if (edgelist_label_offsets || edgelist_hops) {
      compressed_offset_label_hop_offsets = std::move(offset_array_offsets);
    }
  }

  edgelist_hops = std::nullopt;

  return std::make_tuple(
    doubly_compress ? std::make_optional(std::move(compressed_nzd_vertices)) : std::nullopt,
    std::move(compressed_offsets),
    std::move(edgelist_minors),
    std::move(edgelist_weights),
    std::move(edgelist_edge_ids),
    std::move(edgelist_edge_types),
    std::move(compressed_offset_label_hop_offsets),
    std::move(renumber_map),
    std::move(renumber_map_label_offsets));
}

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,       // (label, hop) offsets to the edges
           rmm::device_uvector<vertex_t>,                    // renumber map
           std::optional<rmm::device_uvector<size_t>>>       // label offsets to the renumber map
renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool src_is_major,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;

  auto num_labels = edgelist_label_offsets ? std::get<1>(*edgelist_label_offsets) : size_t{1};
  auto num_hops   = edgelist_hops ? std::get<1>(*edgelist_hops) : size_t{1};

  // 1. check input arguments

  check_input_edges<label_index_t>(handle,
                                   edgelist_srcs,
                                   edgelist_dsts,
                                   edgelist_weights,
                                   edgelist_edge_ids,
                                   edgelist_edge_types,
                                   edgelist_hops,
                                   edgelist_label_offsets,
                                   do_expensive_check);

  // 2. renumber

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{std::nullopt};
  std::tie(edgelist_majors, edgelist_minors, renumber_map, renumber_map_label_offsets) =
    renumber_sampled_edgelist<vertex_t, label_index_t>(
      handle,
      std::move(edgelist_majors),
      std::move(edgelist_minors),
      edgelist_hops ? std::make_optional(std::make_tuple(
                        raft::device_span<int32_t const>(std::get<0>(*edgelist_hops).data(),
                                                         std::get<0>(*edgelist_hops).size()),
                        num_hops))
                    : std::nullopt,
      edgelist_label_offsets,
      do_expensive_check);

  // 3. sort by ((l), (h), major, minor)

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets);

  // 4. compute edgelist_label_hop_offsets

  std::optional<rmm::device_uvector<size_t>> edgelist_label_hop_offsets{std::nullopt};
  if (edgelist_label_offsets || edgelist_hops) {
    edgelist_label_hop_offsets =
      rmm::device_uvector<size_t>(num_labels * num_hops + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 (*edgelist_label_hop_offsets).begin(),
                 (*edgelist_label_hop_offsets).end(),
                 size_t{0});
    // FIXME: the device lambda should be placed in cuda::proclaim_return_type<size_t>()
    // once we update CCCL version to 2.x
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_labels * num_hops),
      (*edgelist_label_hop_offsets).begin(),
      [edgelist_label_offsets = edgelist_label_offsets
                                  ? thrust::make_optional(std::get<0>(*edgelist_label_offsets))
                                  : thrust::nullopt,
       edgelist_hops          = edgelist_hops
                                  ? thrust::make_optional<raft::device_span<int32_t const>>(
                             std::get<0>(*edgelist_hops).data(), std::get<0>(*edgelist_hops).size())
                                  : thrust::nullopt,
       num_hops,
       num_edges = edgelist_majors.size()] __device__(size_t i) {
        size_t start_offset{0};
        auto end_offset = num_edges;

        if (edgelist_label_offsets) {
          auto l_idx   = static_cast<label_index_t>(i / num_hops);
          start_offset = (*edgelist_label_offsets)[l_idx];
          end_offset   = (*edgelist_label_offsets)[l_idx + 1];
        }

        if (edgelist_hops) {
          auto h        = static_cast<int32_t>(i % num_hops);
          auto lower_it = thrust::lower_bound(thrust::seq,
                                              (*edgelist_hops).begin() + start_offset,
                                              (*edgelist_hops).begin() + end_offset,
                                              h);
          auto upper_it = thrust::upper_bound(thrust::seq,
                                              (*edgelist_hops).begin() + start_offset,
                                              (*edgelist_hops).begin() + end_offset,
                                              h);
          start_offset  = static_cast<size_t>(thrust::distance((*edgelist_hops).begin(), lower_it));
          end_offset    = static_cast<size_t>(thrust::distance((*edgelist_hops).begin(), upper_it));
        }

        return end_offset - start_offset;
      });
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           (*edgelist_label_hop_offsets).begin(),
                           (*edgelist_label_hop_offsets).end(),
                           (*edgelist_label_hop_offsets).begin());
  }

  edgelist_hops = std::nullopt;

  return std::make_tuple(std::move(src_is_major ? edgelist_majors : edgelist_minors),
                         std::move(src_is_major ? edgelist_minors : edgelist_majors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_label_hop_offsets),
                         std::move(renumber_map),
                         std::move(renumber_map_label_offsets));
}

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>>       // (label, hop) offsets to the edges
sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool src_is_major,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;

  auto num_labels = edgelist_label_offsets ? std::get<1>(*edgelist_label_offsets) : size_t{1};
  auto num_hops   = edgelist_hops ? std::get<1>(*edgelist_hops) : size_t{1};

  // 1. check input arguments

  check_input_edges<label_index_t>(handle,
                                   edgelist_srcs,
                                   edgelist_dsts,
                                   edgelist_weights,
                                   edgelist_edge_ids,
                                   edgelist_edge_types,
                                   edgelist_hops,
                                   edgelist_label_offsets,
                                   do_expensive_check);

  // 2. sort by ((l), (h), major, minor)

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets);

  // 3. compute edgelist_label_hop_offsets

  std::optional<rmm::device_uvector<size_t>> edgelist_label_hop_offsets{std::nullopt};
  if (edgelist_label_offsets || edgelist_hops) {
    edgelist_label_hop_offsets =
      rmm::device_uvector<size_t>(num_labels * num_hops + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 (*edgelist_label_hop_offsets).begin(),
                 (*edgelist_label_hop_offsets).end(),
                 size_t{0});
    // FIXME: the device lambda should be placed in cuda::proclaim_return_type<size_t>()
    // once we update CCCL version to 2.x
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_labels * num_hops),
      (*edgelist_label_hop_offsets).begin(),
      [edgelist_label_offsets = edgelist_label_offsets
                                  ? thrust::make_optional(std::get<0>(*edgelist_label_offsets))
                                  : thrust::nullopt,
       edgelist_hops          = edgelist_hops
                                  ? thrust::make_optional<raft::device_span<int32_t const>>(
                             std::get<0>(*edgelist_hops).data(), std::get<0>(*edgelist_hops).size())
                                  : thrust::nullopt,
       num_hops,
       num_edges = edgelist_majors.size()] __device__(size_t i) {
        size_t start_offset{0};
        auto end_offset = num_edges;

        if (edgelist_label_offsets) {
          auto l_idx   = static_cast<label_index_t>(i / num_hops);
          start_offset = (*edgelist_label_offsets)[l_idx];
          end_offset   = (*edgelist_label_offsets)[l_idx + 1];
        }

        if (edgelist_hops) {
          auto h        = static_cast<int32_t>(i % num_hops);
          auto lower_it = thrust::lower_bound(thrust::seq,
                                              (*edgelist_hops).begin() + start_offset,
                                              (*edgelist_hops).begin() + end_offset,
                                              h);
          auto upper_it = thrust::upper_bound(thrust::seq,
                                              (*edgelist_hops).begin() + start_offset,
                                              (*edgelist_hops).begin() + end_offset,
                                              h);
          start_offset  = static_cast<size_t>(thrust::distance((*edgelist_hops).begin(), lower_it));
          end_offset    = static_cast<size_t>(thrust::distance((*edgelist_hops).begin(), upper_it));
        }

        return end_offset - start_offset;
      });
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           (*edgelist_label_hop_offsets).begin(),
                           (*edgelist_label_hop_offsets).end(),
                           (*edgelist_label_hop_offsets).begin());
  }

  edgelist_hops = std::nullopt;

  return std::make_tuple(std::move(src_is_major ? edgelist_majors : edgelist_minors),
                         std::move(src_is_major ? edgelist_minors : edgelist_majors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_label_hop_offsets));
}

}  // namespace cugraph

/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "prims/kv_store.cuh"

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

#include <optional>

// FIXME: deprecated, to be deleted
namespace cugraph {

namespace {

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

      auto [h_label_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
        handle, *label_offsets, vertices.size(), approx_edges_to_sort_per_iteration);
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
                     raft::device_span<vertex_t const> edgelist_srcs,
                     raft::device_span<vertex_t const> edgelist_dsts,
                     std::optional<raft::device_span<int32_t const>> edgelist_hops,
                     std::optional<raft::device_span<size_t const>> label_offsets)
{
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    (1 << 20) /* tuning parameter */;  // for segmented sort

  std::optional<rmm::device_uvector<label_index_t>> edgelist_label_indices{std::nullopt};
  if (label_offsets) {
    edgelist_label_indices =
      detail::expand_sparse_offsets(*label_offsets, label_index_t{0}, handle.get_stream());
  }

  auto [unique_label_src_pair_label_indices,
        unique_label_src_pair_vertices,
        unique_label_src_pair_hops,
        unique_label_src_pair_label_offsets] =
    compute_min_hop_for_unique_label_vertex_pairs(
      handle,
      edgelist_srcs,
      edgelist_hops,
      edgelist_label_indices ? std::make_optional<raft::device_span<label_index_t const>>(
                                 (*edgelist_label_indices).data(), (*edgelist_label_indices).size())
                             : std::nullopt,
      label_offsets);

  auto [unique_label_dst_pair_label_indices,
        unique_label_dst_pair_vertices,
        unique_label_dst_pair_hops,
        unique_label_dst_pair_label_offsets] =
    compute_min_hop_for_unique_label_vertex_pairs(
      handle,
      edgelist_dsts,
      edgelist_hops,
      edgelist_label_indices ? std::make_optional<raft::device_span<label_index_t const>>(
                                 (*edgelist_label_indices).data(), (*edgelist_label_indices).size())
                             : std::nullopt,
      label_offsets);

  edgelist_label_indices = std::nullopt;

  if (label_offsets) {
    auto num_labels = (*label_offsets).size() - 1;

    rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
    rmm::device_uvector<label_index_t> renumber_map_label_indices(0, handle.get_stream());

    renumber_map.reserve(
      (*unique_label_src_pair_label_indices).size() + (*unique_label_dst_pair_label_indices).size(),
      handle.get_stream());
    renumber_map_label_indices.reserve(renumber_map.capacity(), handle.get_stream());

    auto num_chunks = (edgelist_srcs.size() + (approx_edges_to_sort_per_iteration - 1)) /
                      approx_edges_to_sort_per_iteration;
    auto chunk_size = (num_chunks > 0) ? ((num_labels + (num_chunks - 1)) / num_chunks) : 0;

    size_t copy_offset{0};
    for (size_t i = 0; i < num_chunks; ++i) {
      auto src_start_offset =
        (*unique_label_src_pair_label_offsets).element(chunk_size * i, handle.get_stream());
      auto src_end_offset =
        (*unique_label_src_pair_label_offsets)
          .element(std::min(chunk_size * (i + 1), num_labels), handle.get_stream());
      auto dst_start_offset =
        (*unique_label_dst_pair_label_offsets).element(chunk_size * i, handle.get_stream());
      auto dst_end_offset =
        (*unique_label_dst_pair_label_offsets)
          .element(std::min(chunk_size * (i + 1), num_labels), handle.get_stream());

      rmm::device_uvector<label_index_t> merged_label_indices(
        (src_end_offset - src_start_offset) + (dst_end_offset - dst_start_offset),
        handle.get_stream());
      rmm::device_uvector<vertex_t> merged_vertices(merged_label_indices.size(),
                                                    handle.get_stream());
      rmm::device_uvector<int8_t> merged_flags(merged_label_indices.size(), handle.get_stream());

      if (edgelist_hops) {
        rmm::device_uvector<int32_t> merged_hops(merged_label_indices.size(), handle.get_stream());
        auto src_quad_first =
          thrust::make_zip_iterator((*unique_label_src_pair_label_indices).begin(),
                                    unique_label_src_pair_vertices.begin(),
                                    (*unique_label_src_pair_hops).begin(),
                                    thrust::make_constant_iterator(int8_t{0}));
        auto dst_quad_first =
          thrust::make_zip_iterator((*unique_label_dst_pair_label_indices).begin(),
                                    unique_label_dst_pair_vertices.begin(),
                                    (*unique_label_dst_pair_hops).begin(),
                                    thrust::make_constant_iterator(int8_t{1}));
        thrust::merge(handle.get_thrust_policy(),
                      src_quad_first + src_start_offset,
                      src_quad_first + src_end_offset,
                      dst_quad_first + dst_start_offset,
                      dst_quad_first + dst_end_offset,
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
        auto src_triplet_first =
          thrust::make_zip_iterator((*unique_label_src_pair_label_indices).begin(),
                                    unique_label_src_pair_vertices.begin(),
                                    thrust::make_constant_iterator(int8_t{0}));
        auto dst_triplet_first =
          thrust::make_zip_iterator((*unique_label_dst_pair_label_indices).begin(),
                                    unique_label_dst_pair_vertices.begin(),
                                    thrust::make_constant_iterator(int8_t{1}));
        thrust::merge(
          handle.get_thrust_policy(),
          src_triplet_first + src_start_offset,
          src_triplet_first + src_end_offset,
          dst_triplet_first + dst_start_offset,
          dst_triplet_first + dst_end_offset,
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
        unique_label_src_pair_vertices.size() + unique_label_dst_pair_vertices.size(),
        handle.get_stream());
      rmm::device_uvector<int32_t> merged_hops(merged_vertices.size(), handle.get_stream());
      rmm::device_uvector<int8_t> merged_flags(merged_vertices.size(), handle.get_stream());
      auto src_triplet_first = thrust::make_zip_iterator(unique_label_src_pair_vertices.begin(),
                                                         (*unique_label_src_pair_hops).begin(),
                                                         thrust::make_constant_iterator(int8_t{0}));
      auto dst_triplet_first = thrust::make_zip_iterator(unique_label_dst_pair_vertices.begin(),
                                                         (*unique_label_dst_pair_hops).begin(),
                                                         thrust::make_constant_iterator(int8_t{1}));
      thrust::merge(handle.get_thrust_policy(),
                    src_triplet_first,
                    src_triplet_first + unique_label_src_pair_vertices.size(),
                    dst_triplet_first,
                    dst_triplet_first + unique_label_dst_pair_vertices.size(),
                    thrust::make_zip_iterator(
                      merged_vertices.begin(), merged_hops.begin(), merged_flags.begin()));

      unique_label_src_pair_vertices.resize(0, handle.get_stream());
      unique_label_src_pair_vertices.shrink_to_fit(handle.get_stream());
      unique_label_src_pair_hops = std::nullopt;
      unique_label_dst_pair_vertices.resize(0, handle.get_stream());
      unique_label_dst_pair_vertices.shrink_to_fit(handle.get_stream());
      unique_label_dst_pair_hops = std::nullopt;

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
      rmm::device_uvector<vertex_t> output_vertices(unique_label_dst_pair_vertices.size(),
                                                    handle.get_stream());
      auto output_last = thrust::set_difference(handle.get_thrust_policy(),
                                                unique_label_dst_pair_vertices.begin(),
                                                unique_label_dst_pair_vertices.end(),
                                                unique_label_src_pair_vertices.begin(),
                                                unique_label_src_pair_vertices.end(),
                                                output_vertices.begin());

      auto num_unique_srcs = unique_label_src_pair_vertices.size();
      auto renumber_map    = std::move(unique_label_src_pair_vertices);
      renumber_map.resize(
        renumber_map.size() + thrust::distance(output_vertices.begin(), output_last),
        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   output_vertices.begin(),
                   output_last,
                   renumber_map.begin() + num_unique_srcs);

      return std::make_tuple(std::move(renumber_map), std::nullopt);
    }
  }
}

}  // namespace

template <typename vertex_t, typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<size_t>>>
renumber_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<size_t const>>>
    label_offsets,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;

  // 1. check input arguments

  CUGRAPH_EXPECTS(!label_offsets || (std::get<0>(*label_offsets).size() <=
                                     std::numeric_limits<label_index_t>::max()),
                  "Invalid input arguments: current implementation assumes that the number of "
                  "unique labels is no larger than std::numeric_limits<uint32_t>::max().");

  CUGRAPH_EXPECTS(
    edgelist_srcs.size() == edgelist_dsts.size(),
    "Invalid input arguments: edgelist_srcs.size() and edgelist_dsts.size() should coincide.");
  CUGRAPH_EXPECTS(!edgelist_hops.has_value() || (edgelist_srcs.size() == (*edgelist_hops).size()),
                  "Invalid input arguments: if edgelist_hops is valid, (*edgelist_hops).size() and "
                  "edgelist_srcs.size() should coincide.");
  CUGRAPH_EXPECTS(!label_offsets.has_value() ||
                    (std::get<1>(*label_offsets).size() == std::get<0>(*label_offsets).size() + 1),
                  "Invalid input arguments: if label_offsets is valid, "
                  "std::get<1>(label_offsets).size() (size of the offset array) should be "
                  "std::get<0>(label_offsets).size() (number of unique labels) + 1.");

  if (do_expensive_check) {
    if (label_offsets) {
      CUGRAPH_EXPECTS(thrust::is_sorted(handle.get_thrust_policy(),
                                        std::get<1>(*label_offsets).begin(),
                                        std::get<1>(*label_offsets).end()),
                      "Invalid input arguments: if label_offsets is valid, "
                      "std::get<1>(*label_offsets) should be sorted.");
      size_t back_element{};
      raft::update_host(
        &back_element,
        std::get<1>(*label_offsets).data() + (std::get<1>(*label_offsets).size() - 1),
        size_t{1},
        handle.get_stream());
      handle.get_stream();
      CUGRAPH_EXPECTS(back_element == edgelist_srcs.size(),
                      "Invalid input arguments: if label_offsets is valid, the last element of "
                      "std::get<1>(*label_offsets) and edgelist_srcs.size() should coincide.");
    }
  }

  // 2. compute renumber_map

  auto [renumber_map, renumber_map_label_indices] = compute_renumber_map<vertex_t, label_index_t>(
    handle,
    raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()),
    raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size()),
    edgelist_hops,
    label_offsets ? std::make_optional<raft::device_span<size_t const>>(std::get<1>(*label_offsets))
                  : std::nullopt);

  // 3. compute renumber map offsets for each label

  std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{};
  if (label_offsets) {
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
      rmm::device_uvector<size_t>(std::get<0>(*label_offsets).size() + 1, handle.get_stream());
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

  // 4. renumber input edges

  if (label_offsets) {
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

    auto num_labels = std::get<0>(*label_offsets).size();

    rmm::device_uvector<vertex_t> segment_sorted_renumber_map(renumber_map.size(),
                                                              handle.get_stream());
    rmm::device_uvector<vertex_t> segment_sorted_new_vertices(new_vertices.size(),
                                                              handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    auto approx_edges_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20) /* tuning parameter */;  // for segmented sort

    auto [h_label_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>{(*renumber_map_label_offsets).data(),
                                      (*renumber_map_label_offsets).size()},
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
      std::get<1>(*label_offsets), label_index_t{0}, handle.get_stream());

    auto pair_first =
      thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_label_indices.begin());
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_srcs.size(),
      edgelist_srcs.begin(),
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

    pair_first = thrust::make_zip_iterator(edgelist_dsts.begin(), edgelist_label_indices.begin());
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_dsts.size(),
      edgelist_dsts.begin(),
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
      edgelist_srcs.begin(), edgelist_srcs.end(), edgelist_srcs.begin(), handle.get_stream());
    kv_store_view.find(
      edgelist_dsts.begin(), edgelist_dsts.end(), edgelist_dsts.begin(), handle.get_stream());
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(renumber_map),
                         std::move(renumber_map_label_offsets));
}

}  // namespace cugraph

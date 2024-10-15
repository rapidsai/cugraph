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

#include "structure/detail/structure_utils.cuh"

#include <cugraph/utilities/dataframe_buffer.hpp>
// FIXME: mem_frugal_partition should probably not be in shuffle_comm.hpp
//        It's used here without any notion of shuffling
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda/std/cstddef>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cuco/hash_functions.cuh>

#include <algorithm>
#include <optional>

namespace cugraph {

namespace detail {

template <typename vertex_t>
struct hash_src_dst_pair {
  int32_t num_groups;

  int32_t __device__ operator()(thrust::tuple<vertex_t, vertex_t> t) const
  {
    vertex_t pair[2];
    pair[0] = thrust::get<0>(t);
    pair[1] = thrust::get<1>(t);
    cuco::murmurhash3_32<vertex_t*> hash_func{};
    return hash_func.compute_hash(reinterpret_cast<cuda::std::byte*>(pair), 2 * sizeof(vertex_t)) %
           num_groups;
  }
};

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> group_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  size_t mem_frugal_threshold)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  if (edgelist_srcs.size() > mem_frugal_threshold) {
    // FIXME: Tuning parameter to address high frequency multi-edges
    //        Defaulting to 2 which makes the code easier.  If
    //        num_groups > 2 we can evaluate whether to find a good
    //        midpoint to do 2 sorts, or if we should do more than 2 sorts.
    const size_t num_groups{2};

    auto group_counts = groupby_and_count(pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          hash_src_dst_pair<vertex_t>{},
                                          num_groups,
                                          mem_frugal_threshold,
                                          handle.get_stream());

    std::vector<size_t> h_group_counts(group_counts.size());
    raft::update_host(
      h_group_counts.data(), group_counts.data(), group_counts.size(), handle.get_stream());

    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + h_group_counts[0]);
    thrust::sort(handle.get_thrust_policy(),
                 pair_first + h_group_counts[0],
                 pair_first + edgelist_srcs.size());
  } else {
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + edgelist_srcs.size());
  }

  return std::make_tuple(std::move(edgelist_srcs), std::move(edgelist_dsts));
}

template <typename vertex_t, typename edge_value_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           decltype(allocate_dataframe_buffer<edge_value_t>(size_t{0}, rmm::cuda_stream_view{}))>
group_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  decltype(allocate_dataframe_buffer<edge_value_t>(0, rmm::cuda_stream_view{}))&& edgelist_values,
  size_t mem_frugal_threshold,
  bool keep_min_value_edge)
{
  auto pair_first  = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
  auto value_first = get_dataframe_buffer_begin(edgelist_values);
  auto edge_first  = thrust::make_zip_iterator(pair_first, value_first);

  if (edgelist_srcs.size() > mem_frugal_threshold) {
    // FIXME: Tuning parameter to address high frequency multi-edges
    //        Defaulting to 2 which makes the code easier.  If
    //        num_groups > 2 we can evaluate whether to find a good
    //        midpoint to do 2 sorts, or if we should do more than 2 sorts.
    const size_t num_groups{2};

    auto group_counts = groupby_and_count(pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          value_first,
                                          hash_src_dst_pair<vertex_t>{},
                                          num_groups,
                                          mem_frugal_threshold,
                                          handle.get_stream());

    std::vector<size_t> h_group_counts(group_counts.size());
    raft::update_host(
      h_group_counts.data(), group_counts.data(), group_counts.size(), handle.get_stream());

    if (keep_min_value_edge) {
      thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + h_group_counts[0]);
      thrust::sort(handle.get_thrust_policy(),
                   edge_first + h_group_counts[0],
                   edge_first + edgelist_srcs.size());
    } else {
      thrust::sort_by_key(
        handle.get_thrust_policy(), pair_first, pair_first + h_group_counts[0], value_first);
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first + h_group_counts[0],
                          pair_first + edgelist_srcs.size(),
                          value_first + h_group_counts[0]);
    }
  } else {
    if (keep_min_value_edge) {
      thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
    } else {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + edgelist_srcs.size(),
                          get_dataframe_buffer_begin(edgelist_values));
    }
  }

  return std::make_tuple(
    std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_values));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<vertex_t>&& edgelist_srcs,
                   rmm::device_uvector<vertex_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge)
{
  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  size_t element_size   = sizeof(vertex_t) * 2;
  if (edgelist_weights) { element_size += sizeof(weight_t); }
  if (edgelist_edge_ids) { element_size += sizeof(edge_t); }
  if (edgelist_edge_types) { element_size += sizeof(edge_type_t); }

  auto constexpr mem_frugal_ratio =
    0.25;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  if (edgelist_weights) {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        std::forward_as_tuple(edgelist_srcs,
                              edgelist_dsts,
                              std::tie(edgelist_weights, edgelist_edge_ids, edgelist_edge_types)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<weight_t, edge_t, edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights),
                            std::move(*edgelist_edge_ids),
                            std::move(*edgelist_edge_types)),
            mem_frugal_threshold,
            keep_min_value_edge);
      } else {
        std::forward_as_tuple(
          edgelist_srcs, edgelist_dsts, std::tie(edgelist_weights, edgelist_edge_ids)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<weight_t, edge_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights), std::move(*edgelist_edge_ids)),
            mem_frugal_threshold,
            keep_min_value_edge);
      }
    } else {
      if (edgelist_edge_types) {
        std::forward_as_tuple(
          edgelist_srcs, edgelist_dsts, std::tie(edgelist_weights, edgelist_edge_types)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<weight_t, edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights), std::move(*edgelist_edge_types)),
            mem_frugal_threshold,
            keep_min_value_edge);
      } else {
        std::forward_as_tuple(edgelist_srcs, edgelist_dsts, std::tie(edgelist_weights)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<weight_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights)),
            mem_frugal_threshold,
            keep_min_value_edge);
      }
    }
  } else {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        std::forward_as_tuple(
          edgelist_srcs, edgelist_dsts, std::tie(edgelist_edge_ids, edgelist_edge_types)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<edge_t, edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_ids), std::move(*edgelist_edge_types)),
            mem_frugal_threshold,
            keep_min_value_edge);
      } else {
        std::forward_as_tuple(edgelist_srcs, edgelist_dsts, std::tie(edgelist_edge_ids)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<edge_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_ids)),
            mem_frugal_threshold,
            keep_min_value_edge);
      }
    } else {
      if (edgelist_edge_types) {
        std::forward_as_tuple(edgelist_srcs, edgelist_dsts, std::tie(edgelist_edge_types)) =
          detail::group_multi_edges<vertex_t, thrust::tuple<edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_types)),
            mem_frugal_threshold,
            keep_min_value_edge);
      } else {
        std::tie(edgelist_srcs, edgelist_dsts) = detail::group_multi_edges(
          handle, std::move(edgelist_srcs), std::move(edgelist_dsts), mem_frugal_threshold);
      }
    }
  }

  auto [keep_count, keep_flags] = detail::mark_entries(
    handle,
    edgelist_srcs.size(),
    [d_edgelist_srcs = edgelist_srcs.data(),
     d_edgelist_dsts = edgelist_dsts.data()] __device__(auto idx) {
      return !((idx > 0) && (d_edgelist_srcs[idx - 1] == d_edgelist_srcs[idx]) &&
               (d_edgelist_dsts[idx - 1] == d_edgelist_dsts[idx]));
    });

  if (keep_count < edgelist_srcs.size()) {
    edgelist_srcs = detail::keep_flagged_elements(
      handle,
      std::move(edgelist_srcs),
      raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
      keep_count);
    edgelist_dsts = detail::keep_flagged_elements(
      handle,
      std::move(edgelist_dsts),
      raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
      keep_count);

    if (edgelist_weights)
      edgelist_weights = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_weights),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

    if (edgelist_edge_ids)
      edgelist_edge_ids = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_edge_ids),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

    if (edgelist_edge_types)
      edgelist_edge_types = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_edge_types),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types));
}

}  // namespace cugraph

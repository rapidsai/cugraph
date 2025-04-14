/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
// FIXME: mem_frugal_partition should probably not be in shuffle_comm.hpp
//        It's used here without any notion of shuffling
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
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
struct hash_src_dst_pair_t {
  using result_type = std::conditional_t<sizeof(vertex_t) == 8,
                                         typename cuco::xxhash_64<vertex_t>::result_type,
                                         typename cuco::xxhash_32<vertex_t>::result_type>;

  __device__ result_type operator()(thrust::tuple<vertex_t, vertex_t> pair) const
  {
    vertex_t buf[2];
    buf[0] = thrust::get<0>(pair);
    buf[1] = thrust::get<1>(pair);
    std::conditional_t<sizeof(vertex_t) == 8, cuco::xxhash_64<vertex_t>, cuco::xxhash_32<vertex_t>>
      hash_func{};
    return hash_func.compute_hash(reinterpret_cast<cuda::std::byte*>(buf), 2 * sizeof(vertex_t));
  }
};

template <typename vertex_t>
struct hash_and_mod_src_dst_pair_t {
  int mod{};

  __device__ int operator()(thrust::tuple<vertex_t, vertex_t> pair) const
  {
    vertex_t buf[2];
    buf[0] = thrust::get<0>(pair);
    buf[1] = thrust::get<1>(pair);
    std::conditional_t<sizeof(vertex_t) == 8, cuco::xxhash_64<vertex_t>, cuco::xxhash_32<vertex_t>>
      hash_func{};
    return static_cast<int>(
      hash_func.compute_hash(reinterpret_cast<cuda::std::byte*>(buf), 2 * sizeof(vertex_t)) % mod);
  }
};

template <typename edge_t, typename weight_t, typename edge_type_t, typename edge_time_t>
struct edge_value_compare_t {
  cuda::std::optional<raft::device_span<weight_t const>> weights;
  cuda::std::optional<raft::device_span<edge_t const>> edge_ids;
  cuda::std::optional<raft::device_span<edge_type_t const>> edge_types;
  cuda::std::optional<raft::device_span<edge_time_t const>> edge_start_times;
  cuda::std::optional<raft::device_span<edge_time_t const>> edge_end_times;

  __device__ bool operator()(size_t l_idx, size_t r_idx) const
  {
    if (weights) {
      auto l_weight = (*weights)[l_idx];
      auto r_weight = (*weights)[r_idx];
      if (l_weight != r_weight) { return l_weight < r_weight; }
    }
    if (edge_ids) {
      auto l_edge_id = (*edge_ids)[l_idx];
      auto r_edge_id = (*edge_ids)[r_idx];
      if (l_edge_id != r_edge_id) { return l_edge_id < r_edge_id; }
    }
    if (edge_types) {
      auto l_edge_type = (*edge_types)[l_idx];
      auto r_edge_type = (*edge_types)[r_idx];
      if (l_edge_type != r_edge_type) { return l_edge_type < r_edge_type; }
    }
    if (edge_start_times) {
      auto l_edge_start_time = (*edge_start_times)[l_idx];
      auto r_edge_start_time = (*edge_start_times)[r_idx];
      if (l_edge_start_time != r_edge_start_time) { return l_edge_start_time < r_edge_start_time; }
    }
    if (edge_end_times) {
      auto l_edge_end_time = (*edge_end_times)[l_idx];
      auto r_edge_end_time = (*edge_end_times)[r_idx];
      if (l_edge_end_time != r_edge_end_time) { return l_edge_end_time < r_edge_end_time; }
    }
    return false;
  }
};

template <typename vertex_t, typename edge_value_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<edge_value_t>>,
           std::vector<size_t>>
group_edges(raft::handle_t const& handle,
            rmm::device_uvector<vertex_t>&& edgelist_srcs,
            rmm::device_uvector<vertex_t>&& edgelist_dsts,
            std::optional<rmm::device_uvector<edge_value_t>>&& edgelist_values,
            int num_groups)
{
  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio =
    0.25;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold = static_cast<size_t>(
    static_cast<double>(total_global_mem / (sizeof(vertex_t) * 2 +
                                            (edgelist_values ? sizeof(edge_value_t) : size_t{0}))) *
    mem_frugal_ratio);

  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
  std::vector<size_t> h_group_counts(num_groups);
  if (edgelist_values) {
    auto d_group_counts = groupby_and_count(pair_first,
                                            pair_first + edgelist_srcs.size(),
                                            edgelist_values->begin(),
                                            hash_and_mod_src_dst_pair_t<vertex_t>{num_groups},
                                            num_groups,
                                            mem_frugal_threshold,
                                            handle.get_stream());
    raft::update_host(
      h_group_counts.data(), d_group_counts.data(), d_group_counts.size(), handle.get_stream());
  } else {
    auto d_group_counts = groupby_and_count(pair_first,
                                            pair_first + edgelist_srcs.size(),
                                            hash_and_mod_src_dst_pair_t<vertex_t>{num_groups},
                                            num_groups,
                                            mem_frugal_threshold,
                                            handle.get_stream());
    raft::update_host(
      h_group_counts.data(), d_group_counts.data(), d_group_counts.size(), handle.get_stream());
  }
  handle.sync_stream();

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_values),
                         std::move(h_group_counts));
}

// returns a device_uvector of bool holding true for multi-edges and false for non-multi-edges
template <typename vertex_t>
std::vector<rmm::device_uvector<bool>> compute_multi_edge_flags(
  raft::handle_t const& handle,
  raft::host_span<raft::device_span<vertex_t const>> edgelist_srcs,
  raft::host_span<raft::device_span<vertex_t const>> edgelist_dsts)
{
  using hash_result_type = typename hash_src_dst_pair_t<vertex_t>::result_type;

  int num_chunks = static_cast<int>(edgelist_srcs.size());

  size_t tot_edges{0};
  for (int i = 0; i < num_chunks; ++i) {
    tot_edges += edgelist_srcs[i].size();
  }

  rmm::device_uvector<hash_result_type> hashes(tot_edges, handle.get_stream());
  {
    size_t offset{0};
    for (int i = 0; i < num_chunks; ++i) {
      auto pair_first =
        thrust::make_zip_iterator(edgelist_srcs[i].begin(), edgelist_dsts[i].begin());
      thrust::transform(handle.get_thrust_policy(),
                        pair_first,
                        pair_first + edgelist_srcs[i].size(),
                        hashes.begin() + offset,
                        hash_src_dst_pair_t<vertex_t>{});
      offset += edgelist_srcs[i].size();
    }
  }

  rmm::device_uvector<hash_result_type> unique_possibly_multi_edge_hashes(0, handle.get_stream());
  size_t num_possibly_multi_edges{0};
  {
    thrust::sort(handle.get_thrust_policy(), hashes.begin(), hashes.end());
    rmm::device_uvector<bool> is_definitely_not_multi_edge_hashes(hashes.size(),
                                                                  handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      is_definitely_not_multi_edge_hashes.begin(),
      is_definitely_not_multi_edge_hashes.end(),
      [hashes = raft::device_span<hash_result_type const>(hashes.data(), hashes.size())] __device__(
        size_t i) {
        if ((i != 0) && (hashes[i] == hashes[i - 1])) {  // compare with the previous element
          return false;
        }
        if ((i != hashes.size() - 1) &&
            (hashes[i] == hashes[i + 1])) {  // compare with the next element
          return false;
        }
        return true;  // definitely unique (src, dst) pair
      });

    num_possibly_multi_edges = thrust::count(handle.get_thrust_policy(),
                                             is_definitely_not_multi_edge_hashes.begin(),
                                             is_definitely_not_multi_edge_hashes.end(),
                                             false);
    unique_possibly_multi_edge_hashes.resize(num_possibly_multi_edges, handle.get_stream());
    thrust::copy_if(
      handle.get_thrust_policy(),
      hashes.begin(),
      hashes.end(),
      is_definitely_not_multi_edge_hashes.begin(),
      unique_possibly_multi_edge_hashes.begin(),
      [] __device__(bool definitely_not_multi_edge) { return !definitely_not_multi_edge; });

    thrust::sort(handle.get_thrust_policy(),
                 unique_possibly_multi_edge_hashes.begin(),
                 unique_possibly_multi_edge_hashes.end());
    unique_possibly_multi_edge_hashes.resize(
      cuda::std::distance(unique_possibly_multi_edge_hashes.begin(),
                          thrust::unique(handle.get_thrust_policy(),
                                         unique_possibly_multi_edge_hashes.begin(),
                                         unique_possibly_multi_edge_hashes.end())),
      handle.get_stream());
  }
  hashes.resize(0, handle.get_stream());
  hashes.shrink_to_fit(handle.get_stream());

  rmm::device_uvector<vertex_t> unique_multi_edge_srcs(num_possibly_multi_edges,
                                                       handle.get_stream());
  rmm::device_uvector<vertex_t> unique_multi_edge_dsts(num_possibly_multi_edges,
                                                       handle.get_stream());
  {
    auto output_pair_first =
      thrust::make_zip_iterator(unique_multi_edge_srcs.begin(), unique_multi_edge_dsts.begin());
    size_t offset = 0;
    for (int i = 0; i < num_chunks; ++i) {
      auto input_pair_first =
        thrust::make_zip_iterator(edgelist_srcs[i].begin(), edgelist_dsts[i].begin());
      auto output_pair_last = thrust::copy_if(
        handle.get_thrust_policy(),
        input_pair_first,
        input_pair_first + edgelist_srcs[i].size(),
        output_pair_first + offset,
        [unique_possibly_multi_edge_hashes = raft::device_span<hash_result_type const>(
           unique_possibly_multi_edge_hashes.data(), unique_possibly_multi_edge_hashes.size()),
         hash_func =
           hash_src_dst_pair_t<vertex_t>{}] __device__(thrust::tuple<vertex_t, vertex_t> pair) {
          auto hash = hash_func(pair);
          return thrust::binary_search(thrust::seq,
                                       unique_possibly_multi_edge_hashes.begin(),
                                       unique_possibly_multi_edge_hashes.end(),
                                       hash);
        });
      offset += cuda::std::distance(output_pair_first + offset, output_pair_last);
    }

    thrust::sort(handle.get_thrust_policy(),
                 output_pair_first,
                 output_pair_first + unique_multi_edge_srcs.size());
    unique_multi_edge_srcs.resize(
      cuda::std::distance(output_pair_first,
                          thrust::unique(handle.get_thrust_policy(),
                                         output_pair_first,
                                         output_pair_first + unique_multi_edge_srcs.size())),
      handle.get_stream());
    unique_multi_edge_dsts.resize(unique_multi_edge_srcs.size(), handle.get_stream());
  }
  unique_possibly_multi_edge_hashes.resize(0, handle.get_stream());
  unique_possibly_multi_edge_hashes.shrink_to_fit(handle.get_stream());

  std::vector<rmm::device_uvector<bool>> multi_edge_flags{};
  multi_edge_flags.reserve(num_chunks);
  for (int i = 0; i < num_chunks; ++i) {
    multi_edge_flags.emplace_back(edgelist_srcs[i].size(), handle.get_stream());
  }

  auto unique_multi_edge_first =
    thrust::make_zip_iterator(unique_multi_edge_srcs.begin(), unique_multi_edge_dsts.begin());
  auto unique_multi_edge_last = unique_multi_edge_first + unique_multi_edge_srcs.size();
  for (int i = 0; i < num_chunks; ++i) {
    auto pair_first = thrust::make_zip_iterator(edgelist_srcs[i].begin(), edgelist_dsts[i].begin());
    thrust::transform(handle.get_thrust_policy(),
                      pair_first,
                      pair_first + edgelist_srcs[i].size(),
                      multi_edge_flags[i].begin(),
                      [unique_multi_edge_first, unique_multi_edge_last] __device__(auto pair) {
                        return thrust::binary_search(
                          thrust::seq, unique_multi_edge_first, unique_multi_edge_last, pair);
                      });
  }

  return multi_edge_flags;
}

template <typename vertex_t, typename edge_value_t>
void sort_multi_edges(raft::handle_t const& handle,
                      raft::device_span<vertex_t> edgelist_srcs,
                      raft::device_span<vertex_t> edgelist_dsts,
                      std::optional<raft::device_span<edge_value_t>> edgelist_values,
                      bool keep_min_value_edge)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
  if (edgelist_values) {
    if (keep_min_value_edge) {
      auto edge_first = thrust::make_zip_iterator(
        edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_values->begin());
      thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
    } else {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + edgelist_srcs.size(),
                          edgelist_values->begin());
    }
  } else {
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + edgelist_srcs.size());
  }

  return;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
void sort_multi_edges(
  raft::handle_t const& handle,
  raft::device_span<vertex_t> edgelist_srcs,
  raft::device_span<vertex_t> edgelist_dsts,
  raft::device_span<size_t> edgelist_indices,
  edge_value_compare_t<edge_t, weight_t, edge_type_t, edge_time_t> edge_value_compare,
  bool keep_min_value_edge)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
  if (keep_min_value_edge) {
    auto edge_compare = [edge_value_compare] __device__(
                          thrust::tuple<vertex_t, vertex_t, size_t> lhs,
                          thrust::tuple<vertex_t, vertex_t, size_t> rhs) {
      auto l_pair = thrust::make_tuple(thrust::get<0>(lhs), thrust::get<1>(lhs));
      auto r_pair = thrust::make_tuple(thrust::get<0>(rhs), thrust::get<1>(rhs));
      if (l_pair == r_pair) {
        return edge_value_compare(thrust::get<2>(lhs), thrust::get<2>(rhs));
      } else {
        return l_pair < r_pair;
      }
    };
    auto edge_first = thrust::make_zip_iterator(
      edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_indices.begin());
    thrust::sort(
      handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size(), edge_compare);
  } else {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        pair_first,
                        pair_first + edgelist_srcs.size(),
                        edgelist_indices.begin());
  }

  return;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<std::vector<rmm::device_uvector<vertex_t>>,
           std::vector<rmm::device_uvector<vertex_t>>,
           std::optional<std::vector<rmm::device_uvector<weight_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_type_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_time_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_time_t>>>>
remove_multi_edges_impl(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_end_times,
  bool keep_min_value_edge)
{
  int edge_property_count = 0;
  if (edgelist_weights) { ++edge_property_count; }
  if (edgelist_edge_ids) { ++edge_property_count; }
  if (edgelist_edge_types) { ++edge_property_count; }
  if (edgelist_edge_start_times) { ++edge_property_count; }
  if (edgelist_edge_end_times) { ++edge_property_count; }

  // 1. group edges (to cut peak memory usage)

  CUGRAPH_EXPECTS(edgelist_srcs.size() <= std::numeric_limits<int>::max(),
                  "Invalid input argument: edgelist_srcs.size() must be less than or equal to "
                  "std::numeric_limits<int>::max().");
  int num_chunks = edgelist_srcs.size();
  int num_groups = std::max(edgelist_srcs.size(), size_t{2});  // to cut peak memory usage

  std::vector<std::vector<size_t>> group_counts(num_chunks);
  std::optional<std::vector<rmm::device_uvector<size_t>>> edgelist_indices{std::nullopt};
  if (edge_property_count > 1) {
    edgelist_indices = std::vector<rmm::device_uvector<size_t>>{};
    for (int i = 0; i < num_chunks; ++i) {
      edgelist_indices->emplace_back(edgelist_srcs[i].size(), handle.get_stream());
    }
  }
  for (int i = 0; i < num_chunks; ++i) {
    if (edge_property_count == 0) {
      std::tie(edgelist_srcs[i], edgelist_dsts[i], std::ignore, group_counts[i]) =
        detail::group_edges<vertex_t, size_t /* dummy */>(handle,
                                                          std::move(edgelist_srcs[i]),
                                                          std::move(edgelist_dsts[i]),
                                                          std::nullopt,
                                                          num_groups);
    } else if (edge_property_count == 1) {
      if (edgelist_weights) {
        std::optional<rmm::device_uvector<weight_t>> tmp{std::nullopt};
        std::tie(edgelist_srcs[i], edgelist_dsts[i], tmp, group_counts[i]) =
          detail::group_edges<vertex_t, weight_t>(
            handle,
            std::move(edgelist_srcs[i]),
            std::move(edgelist_dsts[i]),
            std::make_optional(std::move((*edgelist_weights)[i])),
            num_groups);
        (*edgelist_weights)[i] = std::move(*tmp);
      } else if (edgelist_edge_ids) {
        std::optional<rmm::device_uvector<edge_t>> tmp{std::nullopt};
        std::tie(edgelist_srcs[i], edgelist_dsts[i], tmp, group_counts[i]) =
          detail::group_edges<vertex_t, edge_t>(
            handle,
            std::move(edgelist_srcs[i]),
            std::move(edgelist_dsts[i]),
            std::make_optional(std::move((*edgelist_edge_ids)[i])),
            num_groups);
        (*edgelist_edge_ids)[i] = std::move(*tmp);
      } else if (edgelist_edge_types) {
        std::optional<rmm::device_uvector<edge_type_t>> tmp{std::nullopt};
        std::tie(edgelist_srcs[i], edgelist_dsts[i], tmp, group_counts[i]) =
          detail::group_edges<vertex_t, edge_type_t>(
            handle,
            std::move(edgelist_srcs[i]),
            std::move(edgelist_dsts[i]),
            std::make_optional(std::move((*edgelist_edge_types)[i])),
            num_groups);
        (*edgelist_edge_types)[i] = std::move(*tmp);
      } else if (edgelist_edge_start_times) {
        std::optional<rmm::device_uvector<edge_time_t>> tmp{std::nullopt};
        std::tie(edgelist_srcs[i], edgelist_dsts[i], tmp, group_counts[i]) =
          detail::group_edges<vertex_t, edge_time_t>(
            handle,
            std::move(edgelist_srcs[i]),
            std::move(edgelist_dsts[i]),
            std::make_optional(std::move((*edgelist_edge_start_times)[i])),
            num_groups);
        (*edgelist_edge_start_times)[i] = std::move(*tmp);
      } else {
        assert(edgelist_edge_end_times);
        std::optional<rmm::device_uvector<edge_time_t>> tmp{std::nullopt};
        std::tie(edgelist_srcs[i], edgelist_dsts[i], tmp, group_counts[i]) =
          detail::group_edges<vertex_t, edge_time_t>(
            handle,
            std::move(edgelist_srcs[i]),
            std::move(edgelist_dsts[i]),
            std::make_optional(std::move((*edgelist_edge_end_times)[i])),
            num_groups);
        (*edgelist_edge_end_times)[i] = std::move(*tmp);
      }
    } else {
      detail::sequence_fill(handle.get_stream(),
                            (*edgelist_indices)[i].data(),
                            (*edgelist_indices)[i].size(),
                            size_t{0});
      std::optional<rmm::device_uvector<size_t>> tmp{std::nullopt};
      std::tie(edgelist_srcs[i], edgelist_dsts[i], tmp, group_counts[i]) =
        detail::group_edges<vertex_t, size_t>(handle,
                                              std::move(edgelist_srcs[i]),
                                              std::move(edgelist_dsts[i]),
                                              std::make_optional(std::move((*edgelist_indices)[i])),
                                              num_groups);
      (*edgelist_indices)[i] = std::move(*tmp);
    }
  }

  // 2. for each group, partition non-multi-edges and multi-edges

  std::vector<std::vector<size_t>> group_disps(num_chunks);
  std::vector<std::vector<size_t>> non_multi_edge_counts(num_chunks);
  for (int i = 0; i < num_chunks; ++i) {
    group_disps[i] = std::vector<size_t>(num_groups);
    std::exclusive_scan(
      group_counts[i].begin(), group_counts[i].end(), group_disps[i].begin(), size_t{0});
    non_multi_edge_counts[i] = std::vector<size_t>(num_groups, 0);
  }

  for (int i = 0; i < num_groups; ++i) {
    std::vector<raft::device_span<vertex_t const>> group_edgelist_srcs(num_chunks);
    std::vector<raft::device_span<vertex_t const>> group_edgelist_dsts(num_chunks);
    for (int j = 0; j < num_chunks; ++j) {
      group_edgelist_srcs[j] = raft::device_span<vertex_t const>(
        edgelist_srcs[j].data() + group_disps[j][i], group_counts[j][i]);
      group_edgelist_dsts[j] = raft::device_span<vertex_t const>(
        edgelist_dsts[j].data() + group_disps[j][i], group_counts[j][i]);
    }
    auto multi_edge_flags =
      detail::compute_multi_edge_flags(handle,
                                       raft::host_span<raft::device_span<vertex_t const>>(
                                         group_edgelist_srcs.data(), group_edgelist_srcs.size()),
                                       raft::host_span<raft::device_span<vertex_t const>>(
                                         group_edgelist_dsts.data(), group_edgelist_dsts.size()));

    for (int j = 0; j < num_chunks; ++j) {
      auto pair_first =
        thrust::make_zip_iterator(edgelist_srcs[j].begin(), edgelist_dsts[j].begin()) +
        group_disps[j][i];
      non_multi_edge_counts[j][i] = static_cast<size_t>(
        cuda::std::distance(pair_first,
                            thrust::stable_partition(
                              handle.get_thrust_policy(),
                              pair_first,
                              pair_first + group_counts[j][i],
                              multi_edge_flags[j].begin(),
                              [] __device__(auto multi_edge_flag) { return !multi_edge_flag; })));
      if (edge_property_count == 0) {
        /* nothing to do */
      } else if (edge_property_count == 1) {
        if (edgelist_weights) {
          thrust::stable_partition(
            handle.get_thrust_policy(),
            (*edgelist_weights)[j].begin() + group_disps[j][i],
            (*edgelist_weights)[j].begin() + (group_disps[j][i] + group_counts[j][i]),
            multi_edge_flags[j].begin(),
            [] __device__(auto multi_edge_flag) { return !multi_edge_flag; });
        } else if (edgelist_edge_ids) {
          thrust::stable_partition(
            handle.get_thrust_policy(),
            (*edgelist_edge_ids)[j].begin() + group_disps[j][i],
            (*edgelist_edge_ids)[j].begin() + (group_disps[j][i] + group_counts[j][i]),
            multi_edge_flags[j].begin(),
            [] __device__(auto multi_edge_flag) { return !multi_edge_flag; });
        } else if (edgelist_edge_types) {
          thrust::stable_partition(
            handle.get_thrust_policy(),
            (*edgelist_edge_types)[j].begin() + group_disps[j][i],
            (*edgelist_edge_types)[j].begin() + (group_disps[j][i] + group_counts[j][i]),
            multi_edge_flags[j].begin(),
            [] __device__(auto multi_edge_flag) { return !multi_edge_flag; });
        } else if (edgelist_edge_start_times) {
          thrust::stable_partition(
            handle.get_thrust_policy(),
            (*edgelist_edge_start_times)[j].begin() + group_disps[j][i],
            (*edgelist_edge_start_times)[j].begin() + (group_disps[j][i] + group_counts[j][i]),
            multi_edge_flags[j].begin(),
            [] __device__(auto multi_edge_flag) { return !multi_edge_flag; });
        } else {
          assert(edgelist_edge_end_times);
          thrust::stable_partition(
            handle.get_thrust_policy(),
            (*edgelist_edge_end_times)[j].begin() + group_disps[j][i],
            (*edgelist_edge_end_times)[j].begin() + (group_disps[j][i] + group_counts[j][i]),
            multi_edge_flags[j].begin(),
            [] __device__(auto multi_edge_flag) { return !multi_edge_flag; });
        }
      } else {
        thrust::stable_partition(
          handle.get_thrust_policy(),
          (*edgelist_indices)[j].begin() + group_disps[j][i],
          (*edgelist_indices)[j].begin() + (group_disps[j][i] + group_counts[j][i]),
          multi_edge_flags[j].begin(),
          [] __device__(auto multi_edge_flag) { return !multi_edge_flag; });
      }
    }
  }

  if (num_chunks > 1) {
    std::vector<std::vector<size_t>> group_valid_counts(num_chunks);
    for (int i = 0; i < num_chunks; ++i) {
      group_valid_counts[i] = std::vector<size_t>(num_groups, 0);
    }

    for (int i = 0; i < num_groups; ++i) {
      // 3. for each group, aggregate multi-edges from every chunk to a single multi-edge list

      std::vector<size_t> multi_edge_counts(num_chunks);
      for (int j = 0; j < num_chunks; ++j) {
        multi_edge_counts[j] = group_counts[j][i] - non_multi_edge_counts[j][i];
      }
      std::vector<size_t> multi_edge_disps(num_chunks);
      std::exclusive_scan(
        multi_edge_counts.begin(), multi_edge_counts.end(), multi_edge_disps.begin(), size_t{0});
      auto tot_multi_edge_count = multi_edge_disps.back() + multi_edge_counts.back();

      rmm::device_uvector<vertex_t> multi_edge_srcs(tot_multi_edge_count, handle.get_stream());
      rmm::device_uvector<vertex_t> multi_edge_dsts(tot_multi_edge_count, handle.get_stream());
      auto multi_edge_weights = edgelist_weights ? std::make_optional(rmm::device_uvector<weight_t>(
                                                     tot_multi_edge_count, handle.get_stream()))
                                                 : std::nullopt;
      auto multi_edge_edge_ids = edgelist_edge_ids ? std::make_optional(rmm::device_uvector<edge_t>(
                                                       tot_multi_edge_count, handle.get_stream()))
                                                   : std::nullopt;
      auto multi_edge_edge_types       = edgelist_edge_types
                                           ? std::make_optional(rmm::device_uvector<edge_type_t>(
                                         tot_multi_edge_count, handle.get_stream()))
                                           : std::nullopt;
      auto multi_edge_edge_start_times = edgelist_edge_start_times
                                           ? std::make_optional(rmm::device_uvector<edge_time_t>(
                                               tot_multi_edge_count, handle.get_stream()))
                                           : std::nullopt;
      auto multi_edge_edge_end_times   = edgelist_edge_end_times
                                           ? std::make_optional(rmm::device_uvector<edge_time_t>(
                                             tot_multi_edge_count, handle.get_stream()))
                                           : std::nullopt;
      auto multi_edge_edgelist_indices = edgelist_indices
                                           ? std::make_optional(rmm::device_uvector<size_t>(
                                               tot_multi_edge_count, handle.get_stream()))
                                           : std::nullopt;

      for (int j = 0; j < num_chunks; ++j) {
        auto input_start_offset  = group_disps[j][i] + non_multi_edge_counts[j][i];
        auto input_end_offset    = group_disps[j][i] + group_counts[j][i];
        auto output_start_offset = multi_edge_disps[j];
        thrust::copy(handle.get_thrust_policy(),
                     edgelist_srcs[j].begin() + input_start_offset,
                     edgelist_srcs[j].begin() + input_end_offset,
                     multi_edge_srcs.begin() + output_start_offset);
        thrust::copy(handle.get_thrust_policy(),
                     edgelist_dsts[j].begin() + input_start_offset,
                     edgelist_dsts[j].begin() + input_end_offset,
                     multi_edge_dsts.begin() + output_start_offset);
        if (edgelist_weights) {
          if (edge_property_count == 1) {
            thrust::copy(handle.get_thrust_policy(),
                         (*edgelist_weights)[j].begin() + input_start_offset,
                         (*edgelist_weights)[j].begin() + input_end_offset,
                         multi_edge_weights->begin() + output_start_offset);
          } else {  // edge_property_count > 1
            thrust::gather(handle.get_thrust_policy(),
                           (*edgelist_indices)[j].begin() + input_start_offset,
                           (*edgelist_indices)[j].begin() + input_end_offset,
                           (*edgelist_weights)[j].begin(),
                           multi_edge_weights->begin() + output_start_offset);
          }
        }
        if (edgelist_edge_ids) {
          if (edge_property_count == 1) {
            thrust::copy(handle.get_thrust_policy(),
                         (*edgelist_edge_ids)[j].begin() + input_start_offset,
                         (*edgelist_edge_ids)[j].begin() + input_end_offset,
                         multi_edge_edge_ids->begin() + output_start_offset);
          } else {  // edge_property_count > 1
            thrust::gather(handle.get_thrust_policy(),
                           (*edgelist_indices)[j].begin() + input_start_offset,
                           (*edgelist_indices)[j].begin() + input_end_offset,
                           (*edgelist_edge_ids)[j].begin(),
                           multi_edge_edge_ids->begin() + output_start_offset);
          }
        }
        if (edgelist_edge_types) {
          if (edge_property_count == 1) {
            thrust::copy(handle.get_thrust_policy(),
                         (*edgelist_edge_types)[j].begin() + input_start_offset,
                         (*edgelist_edge_types)[j].begin() + input_end_offset,
                         multi_edge_edge_types->begin() + output_start_offset);
          } else {  // edge_property_count > 1
            thrust::gather(handle.get_thrust_policy(),
                           (*edgelist_indices)[j].begin() + input_start_offset,
                           (*edgelist_indices)[j].begin() + input_end_offset,
                           (*edgelist_edge_types)[j].begin(),
                           multi_edge_edge_types->begin() + output_start_offset);
          }
        }
        if (edgelist_edge_start_times) {
          if (edge_property_count == 1) {
            thrust::copy(handle.get_thrust_policy(),
                         (*edgelist_edge_start_times)[j].begin() + input_start_offset,
                         (*edgelist_edge_start_times)[j].begin() + input_end_offset,
                         multi_edge_edge_start_times->begin() + output_start_offset);
          } else {  // edge_property_count > 1
            thrust::gather(handle.get_thrust_policy(),
                           (*edgelist_indices)[j].begin() + input_start_offset,
                           (*edgelist_indices)[j].begin() + input_end_offset,
                           (*edgelist_edge_start_times)[j].begin(),
                           multi_edge_edge_start_times->begin() + output_start_offset);
          }
        }
        if (edgelist_edge_end_times) {
          if (edge_property_count == 1) {
            thrust::copy(handle.get_thrust_policy(),
                         (*edgelist_edge_end_times)[j].begin() + input_start_offset,
                         (*edgelist_edge_end_times)[j].begin() + input_end_offset,
                         multi_edge_edge_end_times->begin() + output_start_offset);
          } else {  // edge_property_count > 1
            thrust::gather(handle.get_thrust_policy(),
                           (*edgelist_indices)[j].begin() + input_start_offset,
                           (*edgelist_indices)[j].begin() + input_end_offset,
                           (*edgelist_edge_end_times)[j].begin(),
                           multi_edge_edge_end_times->begin() + output_start_offset);
          }
        }
        if (edgelist_indices) {  // edge_property_count > 1
          thrust::copy(handle.get_thrust_policy(),
                       (*edgelist_indices)[j].begin() + input_start_offset,
                       (*edgelist_indices)[j].begin() + input_end_offset,
                       multi_edge_edgelist_indices->begin() + output_start_offset);
        }
      }
      auto multi_edge_indices =
        rmm::device_uvector<size_t>(tot_multi_edge_count, handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(),
                       multi_edge_indices.begin(),
                       multi_edge_indices.end(),
                       size_t{0});

      // 4. for each group, sort the aggregate multi-edge list

      detail::sort_multi_edges<vertex_t, edge_t>(
        handle,
        raft::device_span<vertex_t>(multi_edge_srcs.data(), multi_edge_srcs.size()),
        raft::device_span<vertex_t>(multi_edge_dsts.data(), multi_edge_dsts.size()),
        raft::device_span<size_t>(multi_edge_indices.data(), multi_edge_indices.size()),
        detail::edge_value_compare_t<edge_t, weight_t, edge_type_t, edge_time_t>{
          multi_edge_weights ? cuda::std::make_optional(raft::device_span<weight_t const>(
                                 multi_edge_weights->data(), multi_edge_weights->size()))
                             : cuda::std::nullopt,
          multi_edge_edge_ids ? cuda::std::make_optional(raft::device_span<edge_t const>(
                                  multi_edge_edge_ids->data(), multi_edge_edge_ids->size()))
                              : cuda::std::nullopt,
          multi_edge_edge_types ? cuda::std::make_optional(raft::device_span<edge_type_t const>(
                                    multi_edge_edge_types->data(), multi_edge_edge_types->size()))
                                : cuda::std::nullopt,
          multi_edge_edge_start_times
            ? cuda::std::make_optional(raft::device_span<edge_time_t const>(
                multi_edge_edge_start_times->data(), multi_edge_edge_start_times->size()))
            : cuda::std::nullopt,
          multi_edge_edge_end_times
            ? cuda::std::make_optional(raft::device_span<edge_time_t const>(
                multi_edge_edge_end_times->data(), multi_edge_edge_end_times->size()))
            : cuda::std::nullopt},
        keep_min_value_edge);

      // 5. for each group, remove multi-edges (keep just one edge per multi-edge)

      auto pair_first = thrust::make_zip_iterator(multi_edge_srcs.begin(), multi_edge_dsts.begin());
      auto [keep_count, keep_flags] =
        detail::mark_entries(handle,
                             multi_edge_srcs.size(),
                             detail::is_first_in_run_t<decltype(pair_first)>{pair_first});
      multi_edge_srcs = detail::keep_flagged_elements(
        handle,
        std::move(multi_edge_srcs),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
      multi_edge_dsts = detail::keep_flagged_elements(
        handle,
        std::move(multi_edge_dsts),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
      multi_edge_indices = detail::keep_flagged_elements(
        handle,
        std::move(multi_edge_indices),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

      // 6. copy multi-edges back to the original edgelists

      // 6-1. sort & find chunk boundaries

      thrust::sort_by_key(
        handle.get_thrust_policy(),
        multi_edge_indices.begin(),
        multi_edge_indices.end(),
        thrust::make_zip_iterator(multi_edge_srcs.begin(), multi_edge_dsts.begin()));

      std::vector<size_t> h_lasts(num_chunks);
      std::inclusive_scan(multi_edge_counts.begin(), multi_edge_counts.end(), h_lasts.begin());
      rmm::device_uvector<size_t> d_lasts(num_chunks, handle.get_stream());
      raft::update_device(d_lasts.data(), h_lasts.data(), num_chunks, handle.get_stream());
      rmm::device_uvector<size_t> d_keep_count_offsets(num_chunks + 1, handle.get_stream());
      d_keep_count_offsets.set_element_to_zero_async(0, handle.get_stream());
      thrust::lower_bound(handle.get_thrust_policy(),
                          multi_edge_indices.begin(),
                          multi_edge_indices.end(),
                          d_lasts.begin(),
                          d_lasts.end(),
                          d_keep_count_offsets.begin() + 1);
      std::vector<size_t> h_keep_count_offsets(num_chunks + 1);
      raft::update_host(h_keep_count_offsets.data(),
                        d_keep_count_offsets.data(),
                        num_chunks + 1,
                        handle.get_stream());
      handle.sync_stream();

      // 6-2. copy

      for (int j = 0; j < num_chunks; ++j) {
        thrust::copy(handle.get_thrust_policy(),
                     multi_edge_srcs.begin() + h_keep_count_offsets[j],
                     multi_edge_srcs.begin() + h_keep_count_offsets[j + 1],
                     edgelist_srcs[j].begin() + group_disps[j][i] + non_multi_edge_counts[j][i]);
        thrust::copy(handle.get_thrust_policy(),
                     multi_edge_dsts.begin() + h_keep_count_offsets[j],
                     multi_edge_dsts.begin() + h_keep_count_offsets[j + 1],
                     edgelist_dsts[j].begin() + group_disps[j][i] + non_multi_edge_counts[j][i]);
        if (edge_property_count == 0) {
          /* nothing to do */
        } else if (edge_property_count == 1) {
          if (multi_edge_weights) {
            thrust::gather(
              handle.get_thrust_policy(),
              multi_edge_indices.begin() + h_keep_count_offsets[j],
              multi_edge_indices.begin() + h_keep_count_offsets[j + 1],
              multi_edge_weights->begin(),
              (*edgelist_weights)[j].begin() + group_disps[j][i] + non_multi_edge_counts[j][i]);
          }
          if (multi_edge_edge_ids) {
            thrust::gather(
              handle.get_thrust_policy(),
              multi_edge_indices.begin() + h_keep_count_offsets[j],
              multi_edge_indices.begin() + h_keep_count_offsets[j + 1],
              multi_edge_edge_ids->begin(),
              (*edgelist_edge_ids)[j].begin() + group_disps[j][i] + non_multi_edge_counts[j][i]);
          }
          if (multi_edge_edge_types) {
            thrust::gather(
              handle.get_thrust_policy(),
              multi_edge_indices.begin() + h_keep_count_offsets[j],
              multi_edge_indices.begin() + h_keep_count_offsets[j + 1],
              multi_edge_edge_types->begin(),
              (*edgelist_edge_types)[j].begin() + group_disps[j][i] + non_multi_edge_counts[j][i]);
          }
          if (multi_edge_edge_start_times) {
            thrust::gather(handle.get_thrust_policy(),
                           multi_edge_indices.begin() + h_keep_count_offsets[j],
                           multi_edge_indices.begin() + h_keep_count_offsets[j + 1],
                           multi_edge_edge_start_times->begin(),
                           (*edgelist_edge_start_times)[j].begin() + group_disps[j][i] +
                             non_multi_edge_counts[j][i]);
          }
          if (multi_edge_edge_end_times) {
            thrust::gather(handle.get_thrust_policy(),
                           multi_edge_indices.begin() + h_keep_count_offsets[j],
                           multi_edge_indices.begin() + h_keep_count_offsets[j + 1],
                           multi_edge_edge_end_times->begin(),
                           (*edgelist_edge_end_times)[j].begin() + group_disps[j][i] +
                             non_multi_edge_counts[j][i]);
          }
        } else {  // edge_property_count > 1
          thrust::gather(
            handle.get_thrust_policy(),
            multi_edge_indices.begin() + h_keep_count_offsets[j],
            multi_edge_indices.begin() + h_keep_count_offsets[j + 1],
            multi_edge_edgelist_indices->begin(),
            (*edgelist_indices)[j].begin() + group_disps[j][i] + non_multi_edge_counts[j][i]);
        }
        group_valid_counts[j][i] =
          non_multi_edge_counts[j][i] + (h_keep_count_offsets[j + 1] - h_keep_count_offsets[j]);
      }
    }

    // 7. For each chunk, remove invalid edges

    for (int i = 0; i < num_chunks; ++i) {
      std::vector<size_t> h_lasts(num_chunks);
      std::inclusive_scan(group_counts[i].begin(), group_counts[i].end(), h_lasts.begin());
      rmm::device_uvector<size_t> d_lasts(num_chunks, handle.get_stream());
      raft::update_device(d_lasts.data(), h_lasts.data(), num_chunks, handle.get_stream());
      rmm::device_uvector<size_t> d_group_valid_counts(num_chunks, handle.get_stream());
      raft::update_device(
        d_group_valid_counts.data(), group_valid_counts[i].data(), num_chunks, handle.get_stream());

      auto [keep_count, keep_flags] = detail::mark_entries(
        handle,
        edgelist_srcs[i].size(),
        [lasts              = raft::device_span<size_t const>(d_lasts.data(), d_lasts.size()),
         group_valid_counts = raft::device_span<size_t const>(
           d_group_valid_counts.data(), d_group_valid_counts.size())] __device__(auto i) {
          auto group_idx = cuda::std::distance(
            lasts.begin(), thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), i));
          auto intra_group_idx = i - (group_idx == 0 ? 0 : lasts[group_idx - 1]);
          return intra_group_idx < group_valid_counts[group_idx];
        });

      edgelist_srcs[i] = detail::keep_flagged_elements(
        handle,
        std::move(edgelist_srcs[i]),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
      edgelist_dsts[i] = detail::keep_flagged_elements(
        handle,
        std::move(edgelist_dsts[i]),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
      if (edge_property_count == 0) {
        /* nothing to do */
      } else if (edge_property_count == 1) {
        if (edgelist_weights) {
          (*edgelist_weights)[i] = detail::keep_flagged_elements(
            handle,
            std::move((*edgelist_weights)[i]),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
        if (edgelist_edge_ids) {
          (*edgelist_edge_ids)[i] = detail::keep_flagged_elements(
            handle,
            std::move((*edgelist_edge_ids)[i]),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
        if (edgelist_edge_types) {
          (*edgelist_edge_types)[i] = detail::keep_flagged_elements(
            handle,
            std::move((*edgelist_edge_types)[i]),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
        if (edgelist_edge_start_times) {
          (*edgelist_edge_start_times)[i] = detail::keep_flagged_elements(
            handle,
            std::move((*edgelist_edge_start_times)[i]),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
        if (edgelist_edge_end_times) {
          (*edgelist_edge_end_times)[i] = detail::keep_flagged_elements(
            handle,
            std::move((*edgelist_edge_end_times)[i]),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
      } else {  // edge_property_count > 1
        (*edgelist_indices)[i] = detail::keep_flagged_elements(
          handle,
          std::move((*edgelist_indices)[i]),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
        if (edgelist_weights) {
          rmm::device_uvector<weight_t> tmp(keep_count, handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*edgelist_indices)[i].begin(),
                         (*edgelist_indices)[i].begin() + keep_count,
                         (*edgelist_weights)[i].begin(),
                         tmp.begin());
          (*edgelist_weights)[i] = std::move(tmp);
        }
        if (edgelist_edge_ids) {
          rmm::device_uvector<edge_t> tmp(keep_count, handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*edgelist_indices)[i].begin(),
                         (*edgelist_indices)[i].begin() + keep_count,
                         (*edgelist_edge_ids)[i].begin(),
                         tmp.begin());
          (*edgelist_edge_ids)[i] = std::move(tmp);
        }
        if (edgelist_edge_types) {
          rmm::device_uvector<edge_type_t> tmp(keep_count, handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*edgelist_indices)[i].begin(),
                         (*edgelist_indices)[i].begin() + keep_count,
                         (*edgelist_edge_types)[i].begin(),
                         tmp.begin());
          (*edgelist_edge_types)[i] = std::move(tmp);
        }
        if (edgelist_edge_start_times) {
          rmm::device_uvector<edge_time_t> tmp(keep_count, handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*edgelist_indices)[i].begin(),
                         (*edgelist_indices)[i].begin() + keep_count,
                         (*edgelist_edge_start_times)[i].begin(),
                         tmp.begin());
          (*edgelist_edge_start_times)[i] = std::move(tmp);
        }
        if (edgelist_edge_end_times) {
          rmm::device_uvector<edge_time_t> tmp(keep_count, handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*edgelist_indices)[i].begin(),
                         (*edgelist_indices)[i].begin() + keep_count,
                         (*edgelist_edge_end_times)[i].begin(),
                         tmp.begin());
          (*edgelist_edge_end_times)[i] = std::move(tmp);
        }
      }
    }
  } else {  // num_chunks == 1
    // 3. for each group, sort edges

    for (int i = 0; i < num_groups; ++i) {
      if (edge_property_count == 0) {
        detail::sort_multi_edges<vertex_t, size_t /* dummy */>(
          handle,
          raft::device_span<vertex_t>(
            edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
            group_counts[0][i] - non_multi_edge_counts[0][i]),
          raft::device_span<vertex_t>(
            edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
            group_counts[0][i] - non_multi_edge_counts[0][i]),
          std::nullopt,
          keep_min_value_edge);
      } else if (edge_property_count == 1) {
        if (edgelist_weights) {
          detail::sort_multi_edges<vertex_t, weight_t>(
            handle,
            raft::device_span<vertex_t>(
              edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            raft::device_span<vertex_t>(
              edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            std::make_optional(raft::device_span<weight_t>(
              (*edgelist_weights)[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i])),
            keep_min_value_edge);
        } else if (edgelist_edge_ids) {
          detail::sort_multi_edges<vertex_t, edge_t>(
            handle,
            raft::device_span<vertex_t>(
              edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            raft::device_span<vertex_t>(
              edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            std::make_optional(raft::device_span<edge_t>(
              (*edgelist_edge_ids)[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i])),
            keep_min_value_edge);
        } else if (edgelist_edge_types) {
          detail::sort_multi_edges<vertex_t, edge_type_t>(
            handle,
            raft::device_span<vertex_t>(
              edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            raft::device_span<vertex_t>(
              edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            std::make_optional(raft::device_span<edge_type_t>(
              (*edgelist_edge_types)[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i])),
            keep_min_value_edge);
        } else if (edgelist_edge_start_times) {
          detail::sort_multi_edges<vertex_t, edge_time_t>(
            handle,
            raft::device_span<vertex_t>(
              edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            raft::device_span<vertex_t>(
              edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            std::make_optional(
              raft::device_span<edge_time_t>((*edgelist_edge_start_times)[0].data() +
                                               group_disps[0][i] + non_multi_edge_counts[0][i],
                                             group_counts[0][i] - non_multi_edge_counts[0][i])),
            keep_min_value_edge);
        } else {
          assert(edgelist_edge_end_times);
          detail::sort_multi_edges<vertex_t, edge_time_t>(
            handle,
            raft::device_span<vertex_t>(
              edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            raft::device_span<vertex_t>(
              edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
              group_counts[0][i] - non_multi_edge_counts[0][i]),
            std::make_optional(
              raft::device_span<edge_time_t>((*edgelist_edge_end_times)[0].data() +
                                               group_disps[0][i] + non_multi_edge_counts[0][i],
                                             group_counts[0][i] - non_multi_edge_counts[0][i])),
            keep_min_value_edge);
        }
      } else {  // edge_property_count > 1
        detail::sort_multi_edges<vertex_t, edge_t>(
          handle,
          raft::device_span<vertex_t>(
            edgelist_srcs[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
            group_counts[0][i] - non_multi_edge_counts[0][i]),
          raft::device_span<vertex_t>(
            edgelist_dsts[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
            group_counts[0][i] - non_multi_edge_counts[0][i]),
          raft::device_span<size_t>(
            (*edgelist_indices)[0].data() + group_disps[0][i] + non_multi_edge_counts[0][i],
            group_counts[0][i] - non_multi_edge_counts[0][i]),
          detail::edge_value_compare_t<edge_t, weight_t, edge_type_t, edge_time_t>{
            edgelist_weights ? cuda::std::make_optional(raft::device_span<weight_t const>(
                                 (*edgelist_weights)[0].data(), (*edgelist_weights)[0].size()))
                             : cuda::std::nullopt,
            edgelist_edge_ids ? cuda::std::make_optional(raft::device_span<edge_t const>(
                                  (*edgelist_edge_ids)[0].data(), (*edgelist_edge_ids)[0].size()))
                              : cuda::std::nullopt,
            edgelist_edge_types
              ? cuda::std::make_optional(raft::device_span<edge_type_t const>(
                  (*edgelist_edge_types)[0].data(), (*edgelist_edge_types)[0].size()))
              : cuda::std::nullopt,
            edgelist_edge_start_times
              ? cuda::std::make_optional(raft::device_span<edge_time_t const>(
                  (*edgelist_edge_start_times)[0].data(), (*edgelist_edge_start_times)[0].size()))
              : cuda::std::nullopt,
            edgelist_edge_end_times
              ? cuda::std::make_optional(raft::device_span<edge_time_t const>(
                  (*edgelist_edge_end_times)[0].data(), (*edgelist_edge_end_times)[0].size()))
              : cuda::std::nullopt},
          keep_min_value_edge);
      }
    }

    // 4. remove multi-edges (keep just one edge per multi-edge)

    auto pair_first = thrust::make_zip_iterator(edgelist_srcs[0].begin(), edgelist_dsts[0].begin());
    auto [keep_count, keep_flags] = detail::mark_entries(
      handle, edgelist_srcs[0].size(), detail::is_first_in_run_t<decltype(pair_first)>{pair_first});

    edgelist_srcs[0] = detail::keep_flagged_elements(
      handle,
      std::move(edgelist_srcs[0]),
      raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
      keep_count);
    edgelist_dsts[0] = detail::keep_flagged_elements(
      handle,
      std::move(edgelist_dsts[0]),
      raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
      keep_count);

    if (edge_property_count == 0) {
      /* nothing to do */
    } else if (edge_property_count == 1) {
      if (edgelist_weights) {
        (*edgelist_weights)[0] = detail::keep_flagged_elements(
          handle,
          std::move((*edgelist_weights)[0]),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
      }

      if (edgelist_edge_ids) {
        (*edgelist_edge_ids)[0] = detail::keep_flagged_elements(
          handle,
          std::move((*edgelist_edge_ids)[0]),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
      }

      if (edgelist_edge_types) {
        (*edgelist_edge_types)[0] = detail::keep_flagged_elements(
          handle,
          std::move((*edgelist_edge_types)[0]),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
      }

      if (edgelist_edge_start_times) {
        (*edgelist_edge_start_times)[0] = detail::keep_flagged_elements(
          handle,
          std::move((*edgelist_edge_start_times)[0]),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
      }

      if (edgelist_edge_end_times) {
        (*edgelist_edge_end_times)[0] = detail::keep_flagged_elements(
          handle,
          std::move((*edgelist_edge_end_times)[0]),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
      }
    } else {  // edge_property_count > 1
      assert(edgelist_indices);
      (*edgelist_indices)[0] = detail::keep_flagged_elements(
        handle,
        std::move((*edgelist_indices)[0]),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
      if (edgelist_weights) {
        auto tmp = rmm::device_uvector<weight_t>(keep_count, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       (*edgelist_indices)[0].begin(),
                       (*edgelist_indices)[0].end(),
                       (*edgelist_weights)[0].begin(),
                       tmp.begin());
        (*edgelist_weights)[0] = std::move(tmp);
      }
      if (edgelist_edge_ids) {
        auto tmp = rmm::device_uvector<edge_t>(keep_count, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       (*edgelist_indices)[0].begin(),
                       (*edgelist_indices)[0].end(),
                       (*edgelist_edge_ids)[0].begin(),
                       tmp.begin());
        (*edgelist_edge_ids)[0] = std::move(tmp);
      }
      if (edgelist_edge_types) {
        auto tmp = rmm::device_uvector<edge_type_t>(keep_count, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       (*edgelist_indices)[0].begin(),
                       (*edgelist_indices)[0].end(),
                       (*edgelist_edge_types)[0].begin(),
                       tmp.begin());
        (*edgelist_edge_types)[0] = std::move(tmp);
      }
      if (edgelist_edge_start_times) {
        auto tmp = rmm::device_uvector<edge_time_t>(keep_count, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       (*edgelist_indices)[0].begin(),
                       (*edgelist_indices)[0].end(),
                       (*edgelist_edge_start_times)[0].begin(),
                       tmp.begin());
        (*edgelist_edge_start_times)[0] = std::move(tmp);
      }
      if (edgelist_edge_end_times) {
        auto tmp = rmm::device_uvector<edge_time_t>(keep_count, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       (*edgelist_indices)[0].begin(),
                       (*edgelist_indices)[0].end(),
                       (*edgelist_edge_end_times)[0].begin(),
                       tmp.begin());
        (*edgelist_edge_end_times)[0] = std::move(tmp);
      }
    }
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_edge_start_times),
                         std::move(edgelist_edge_end_times));
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<vertex_t>&& edgelist_srcs,
                   rmm::device_uvector<vertex_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                   std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_start_times,
                   std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_end_times,
                   bool keep_min_value_edge)
{
  CUGRAPH_EXPECTS(
    edgelist_dsts.size() == edgelist_srcs.size(),
    "Invalid input argument: edgelist_srcs and edgelist_dsts must have the same size.");
  CUGRAPH_EXPECTS(!edgelist_weights || edgelist_weights->size() == edgelist_srcs.size(),
                  "Invalid input argument: edgelist_weights (if valid) and edgelist_srcs must have "
                  "the same size.");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || edgelist_edge_ids->size() == edgelist_srcs.size(),
                  "Invalid input argument: edgelist_edge_ids (if valid) and edgelist_srcs must "
                  "have the same size.");
  CUGRAPH_EXPECTS(!edgelist_edge_types || edgelist_edge_types->size() == edgelist_srcs.size(),
                  "Invalid input argument: edgelist_edge_types (if valid) and edgelist_srcs must "
                  "have the same size.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_start_times || edgelist_edge_start_times->size() == edgelist_srcs.size(),
    "Invalid input argument: edgelist_edge_start_times (if valid) and edgelist_srcs must have the "
    "same size.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_end_times || edgelist_edge_end_times->size() == edgelist_srcs.size(),
    "Invalid input argument: edgelist_edge_end_times (if valid) and edgelist_srcs must have the "
    "same size.");

  int edge_property_count = 0;
  if (edgelist_weights) { ++edge_property_count; }
  if (edgelist_edge_ids) { ++edge_property_count; }
  if (edgelist_edge_types) { ++edge_property_count; }
  if (edgelist_edge_start_times) { ++edge_property_count; }
  if (edgelist_edge_end_times) { ++edge_property_count; }

  std::vector<rmm::device_uvector<vertex_t>> edgelist_src_chunks{};
  edgelist_src_chunks.push_back(std::move(edgelist_srcs));
  std::vector<rmm::device_uvector<vertex_t>> edgelist_dst_chunks{};
  edgelist_dst_chunks.push_back(std::move(edgelist_dsts));
  std::optional<std::vector<rmm::device_uvector<weight_t>>> edgelist_weight_chunks{std::nullopt};
  if (edgelist_weights) {
    edgelist_weight_chunks = std::vector<rmm::device_uvector<weight_t>>{};
    edgelist_weight_chunks->push_back(std::move(*edgelist_weights));
  }
  std::optional<std::vector<rmm::device_uvector<edge_t>>> edgelist_edge_id_chunks{std::nullopt};
  if (edgelist_edge_ids) {
    edgelist_edge_id_chunks = std::vector<rmm::device_uvector<edge_t>>{};
    edgelist_edge_id_chunks->push_back(std::move(*edgelist_edge_ids));
  }
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edgelist_edge_type_chunks{
    std::nullopt};
  if (edgelist_edge_types) {
    edgelist_edge_type_chunks = std::vector<rmm::device_uvector<edge_type_t>>{};
    edgelist_edge_type_chunks->push_back(std::move(*edgelist_edge_types));
  }
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edgelist_edge_start_time_chunks{
    std::nullopt};
  if (edgelist_edge_start_times) {
    edgelist_edge_start_time_chunks = std::vector<rmm::device_uvector<edge_time_t>>{};
    edgelist_edge_start_time_chunks->push_back(std::move(*edgelist_edge_start_times));
  }
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edgelist_edge_end_time_chunks{
    std::nullopt};
  if (edgelist_edge_end_times) {
    edgelist_edge_end_time_chunks = std::vector<rmm::device_uvector<edge_time_t>>{};
    edgelist_edge_end_time_chunks->push_back(std::move(*edgelist_edge_end_times));
  }

  std::tie(edgelist_src_chunks,
           edgelist_dst_chunks,
           edgelist_weight_chunks,
           edgelist_edge_id_chunks,
           edgelist_edge_type_chunks,
           edgelist_edge_start_time_chunks,
           edgelist_edge_end_time_chunks) =
    detail::remove_multi_edges_impl(handle,
                                    std::move(edgelist_src_chunks),
                                    std::move(edgelist_dst_chunks),
                                    std::move(edgelist_weight_chunks),
                                    std::move(edgelist_edge_id_chunks),
                                    std::move(edgelist_edge_type_chunks),
                                    std::move(edgelist_edge_start_time_chunks),
                                    std::move(edgelist_edge_end_time_chunks),
                                    keep_min_value_edge);

  return std::make_tuple(
    std::move(edgelist_src_chunks[0]),
    std::move(edgelist_dst_chunks[0]),
    edgelist_weight_chunks ? std::make_optional(std::move((*edgelist_weight_chunks)[0]))
                           : std::nullopt,
    edgelist_edge_id_chunks ? std::make_optional(std::move((*edgelist_edge_id_chunks)[0]))
                            : std::nullopt,
    edgelist_edge_type_chunks ? std::make_optional(std::move((*edgelist_edge_type_chunks)[0]))
                              : std::nullopt,
    edgelist_edge_start_time_chunks
      ? std::make_optional(std::move((*edgelist_edge_start_time_chunks)[0]))
      : std::nullopt,
    edgelist_edge_end_time_chunks
      ? std::make_optional(std::move((*edgelist_edge_end_time_chunks)[0]))
      : std::nullopt);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<std::vector<rmm::device_uvector<vertex_t>>,
           std::vector<rmm::device_uvector<vertex_t>>,
           std::optional<std::vector<rmm::device_uvector<weight_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_type_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_time_t>>>,
           std::optional<std::vector<rmm::device_uvector<edge_time_t>>>>
remove_multi_edges(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_end_times,
  bool keep_min_value_edge)
{
  CUGRAPH_EXPECTS(
    edgelist_dsts.size() == edgelist_srcs.size(),
    "Invalid input argument: edgelist_dsts and edgelist_srcs must have the same size.");
  CUGRAPH_EXPECTS(!edgelist_weights || edgelist_weights->size() == edgelist_srcs.size(),
                  "Invalid input argument: edgelist_weights (if valid) and edgelist_edge_ids must "
                  "have the same size.");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || edgelist_edge_ids->size() == edgelist_srcs.size(),
                  "Invalid input argument: edgelist_edge_ids (if valid) and edgelist_srcs must "
                  "have the same size.");
  CUGRAPH_EXPECTS(!edgelist_edge_types || edgelist_edge_types->size() == edgelist_srcs.size(),
                  "Invalid input argument: edgelist_edge_types (if valid) and edgelist_srcs must "
                  "have the same size.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_start_times || edgelist_edge_start_times->size() == edgelist_srcs.size(),
    "Invalid input argument: edgelist_edge_start_times (if valid) and edgelist_srcs must have the "
    "same size.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_end_times || edgelist_edge_end_times->size() == edgelist_srcs.size(),
    "Invalid input argument: edgelist_edge_end_times (if valid) and edgelist_srcs must have the "
    "same size.");
  for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
    CUGRAPH_EXPECTS(
      edgelist_dsts[i].size() == edgelist_srcs[i].size(),
      "Invalid input argument: edgelist_dsts[i] and edgelist_srcs[i] must have the same size.");
    CUGRAPH_EXPECTS(!edgelist_weights || (*edgelist_weights)[i].size() == edgelist_srcs[i].size(),
                    "Invalid input argument: edgelist_weights[i] (if valid) and edgelist_srcs[i] "
                    "must have the same size.");
    CUGRAPH_EXPECTS(!edgelist_edge_ids || (*edgelist_edge_ids)[i].size() == edgelist_srcs[i].size(),
                    "Invalid input argument: edgelist_edge_ids[i] (if valid) and edgelist_srcs[i] "
                    "must have the same size.");
    CUGRAPH_EXPECTS(
      !edgelist_edge_types || (*edgelist_edge_types)[i].size() == edgelist_srcs[i].size(),
      "Invalid input argument: edgelist_edge_types[i] (if valid) and edgelist_srcs[i] must have "
      "the same size.");
    CUGRAPH_EXPECTS(!edgelist_edge_start_times ||
                      (*edgelist_edge_start_times)[i].size() == edgelist_srcs[i].size(),
                    "Invalid input argument: edgelist_edge_start_times[i] (if valid) and "
                    "edgelist_srcs[i] must have the same size.");
    CUGRAPH_EXPECTS(
      !edgelist_edge_end_times || (*edgelist_edge_end_times)[i].size() == edgelist_srcs[i].size(),
      "Invalid input argument: edgelist_edge_end_times[i] (if valid) and edgelist_srcs[i] must "
      "have the same size.");
  }

  return detail::remove_multi_edges_impl(handle,
                                         std::move(edgelist_srcs),
                                         std::move(edgelist_dsts),
                                         std::move(edgelist_weights),
                                         std::move(edgelist_edge_ids),
                                         std::move(edgelist_edge_types),
                                         std::move(edgelist_edge_start_times),
                                         std::move(edgelist_edge_end_times),
                                         keep_min_value_edge);
}

}  // namespace cugraph

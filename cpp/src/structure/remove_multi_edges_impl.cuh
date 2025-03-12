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

template <typename edge_t, typename weight_t, typename edge_type_t, typename edge_time_t>
struct edge_value_compare_t {
  cuda::std::optional<raft::device_span<weight_t const>> weights;
  cuda::std::optional<raft::device_span<edge_t const>> edge_ids;
  cuda::std::optional<raft::device_span<edge_type_t const>> edge_types;
  cuda::std::optional<raft::device_span<edge_time_t const>> edge_start_times;
  cuda::std::optional<raft::device_span<edge_time_t const>> edge_end_times;

  __device__ bool operator()(edge_t l_idx, edge_t r_idx) const
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
           rmm::device_uvector<edge_value_t>>
group_multi_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edgelist_srcs,
                  rmm::device_uvector<vertex_t>&& edgelist_dsts,
                  rmm::device_uvector<edge_value_t>&& edgelist_values,
                  size_t mem_frugal_threshold,
                  bool keep_min_value_edge)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
  auto edge_first = thrust::make_zip_iterator(
    edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_values.begin());

  if (edgelist_srcs.size() > mem_frugal_threshold) {
    // FIXME: Tuning parameter to address high frequency multi-edges
    //        Defaulting to 2 which makes the code easier.  If
    //        num_groups > 2 we can evaluate whether to find a good
    //        midpoint to do 2 sorts, or if we should do more than 2 sorts.
    const size_t num_groups{2};

    auto group_counts = groupby_and_count(pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          edgelist_values.begin(),
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
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + h_group_counts[0],
                          edgelist_values.begin());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first + h_group_counts[0],
                          pair_first + edgelist_srcs.size(),
                          edgelist_values.begin() + h_group_counts[0]);
    }
  } else {
    if (keep_min_value_edge) {
      thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
    } else {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + edgelist_srcs.size(),
                          edgelist_values.begin());
    }
  }

  return std::make_tuple(
    std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_values));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>>
  group_multi_edges(
    raft::handle_t const& handle,
    rmm::device_uvector<vertex_t>&& edgelist_srcs,
    rmm::device_uvector<vertex_t>&& edgelist_dsts,
    rmm::device_uvector<edge_t>&& edgelist_indices,
    edge_value_compare_t<edge_t, weight_t, edge_type_t, edge_time_t> edge_value_compare,
    size_t mem_frugal_threshold,
    bool keep_min_value_edge)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());
  auto edge_first = thrust::make_zip_iterator(
    edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_indices.begin());

  auto edge_compare = [edge_value_compare] __device__(
                        thrust::tuple<vertex_t, vertex_t, edge_t> lhs,
                        thrust::tuple<vertex_t, vertex_t, edge_t> rhs) {
    auto l_pair = thrust::make_tuple(thrust::get<0>(lhs), thrust::get<1>(lhs));
    auto r_pair = thrust::make_tuple(thrust::get<0>(rhs), thrust::get<1>(rhs));
    if (l_pair == r_pair) {
      return edge_value_compare(thrust::get<2>(lhs), thrust::get<2>(rhs));
    } else {
      return l_pair < r_pair;
    }
  };

  if (edgelist_srcs.size() > mem_frugal_threshold) {
    // FIXME: Tuning parameter to address high frequency multi-edges
    //        Defaulting to 2 which makes the code easier.  If
    //        num_groups > 2 we can evaluate whether to find a good
    //        midpoint to do 2 sorts, or if we should do more than 2 sorts.
    const size_t num_groups{2};

    auto group_counts = groupby_and_count(pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          edgelist_indices.begin(),
                                          hash_src_dst_pair<vertex_t>{},
                                          num_groups,
                                          mem_frugal_threshold,
                                          handle.get_stream());

    std::vector<size_t> h_group_counts(group_counts.size());
    raft::update_host(
      h_group_counts.data(), group_counts.data(), group_counts.size(), handle.get_stream());

    if (keep_min_value_edge) {
      thrust::sort(
        handle.get_thrust_policy(), edge_first, edge_first + h_group_counts[0], edge_compare);
      thrust::sort(handle.get_thrust_policy(),
                   edge_first + h_group_counts[0],
                   edge_first + edgelist_srcs.size(),
                   edge_compare);
    } else {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + h_group_counts[0],
                          edgelist_indices.begin());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first + h_group_counts[0],
                          pair_first + edgelist_srcs.size(),
                          edgelist_indices.begin() + h_group_counts[0]);
    }
  } else {
    if (keep_min_value_edge) {
      thrust::sort(
        handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size(), edge_compare);
    } else {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + edgelist_srcs.size(),
                          edgelist_indices.begin());
    }
  }

  return std::make_tuple(
    std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_indices));
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
  auto total_global_mem   = handle.get_device_properties().totalGlobalMem;
  int edge_property_count = 0;
  size_t element_size     = sizeof(vertex_t) * 2;

  if (edgelist_weights) {
    ++edge_property_count;
    element_size += sizeof(weight_t);
  }

  if (edgelist_edge_ids) {
    ++edge_property_count;
    element_size += sizeof(edge_t);
  }
  if (edgelist_edge_types) {
    ++edge_property_count;
    element_size += sizeof(edge_type_t);
  }
  if (edgelist_edge_start_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }
  if (edgelist_edge_end_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }

  if (edge_property_count > 1) { element_size = sizeof(vertex_t) * 2 + sizeof(size_t); }

  auto constexpr mem_frugal_ratio =
    0.25;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  if (edge_property_count == 0) {
    std::tie(edgelist_srcs, edgelist_dsts) = detail::group_multi_edges(
      handle, std::move(edgelist_srcs), std::move(edgelist_dsts), mem_frugal_threshold);
  } else if (edge_property_count == 1) {
    if (edgelist_weights) {
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
        detail::group_multi_edges<vertex_t, weight_t>(handle,
                                                      std::move(edgelist_srcs),
                                                      std::move(edgelist_dsts),
                                                      std::move(*edgelist_weights),
                                                      mem_frugal_threshold,
                                                      keep_min_value_edge);
    } else if (edgelist_edge_ids) {
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_edge_ids) =
        detail::group_multi_edges<vertex_t, edge_t>(handle,
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(*edgelist_edge_ids),
                                                    mem_frugal_threshold,
                                                    keep_min_value_edge);
    } else if (edgelist_edge_types) {
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_edge_types) =
        detail::group_multi_edges<vertex_t, edge_type_t>(handle,
                                                         std::move(edgelist_srcs),
                                                         std::move(edgelist_dsts),
                                                         std::move(*edgelist_edge_types),
                                                         mem_frugal_threshold,
                                                         keep_min_value_edge);
    } else if (edgelist_edge_start_times) {
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_edge_start_times) =
        detail::group_multi_edges<vertex_t, edge_time_t>(handle,
                                                         std::move(edgelist_srcs),
                                                         std::move(edgelist_dsts),
                                                         std::move(*edgelist_edge_start_times),
                                                         mem_frugal_threshold,
                                                         keep_min_value_edge);
    } else if (edgelist_edge_end_times) {
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_edge_end_times) =
        detail::group_multi_edges<vertex_t, edge_time_t>(handle,
                                                         std::move(edgelist_srcs),
                                                         std::move(edgelist_dsts),
                                                         std::move(*edgelist_edge_end_times),
                                                         mem_frugal_threshold,
                                                         keep_min_value_edge);
    }
  } else {
    rmm::device_uvector<edge_t> property_position(edgelist_srcs.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), edge_t{0});

    std::tie(edgelist_srcs, edgelist_dsts, property_position) =
      detail::group_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
        handle,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(property_position),
        detail::edge_value_compare_t<edge_t, weight_t, edge_type_t, edge_time_t>{
          edgelist_weights ? cuda::std::make_optional(raft::device_span<weight_t const>(
                               (*edgelist_weights).data(), (*edgelist_weights).size()))
                           : cuda::std::nullopt,
          edgelist_edge_ids ? cuda::std::make_optional(raft::device_span<edge_t const>(
                                (*edgelist_edge_ids).data(), (*edgelist_edge_ids).size()))
                            : cuda::std::nullopt,
          edgelist_edge_types ? cuda::std::make_optional(raft::device_span<edge_type_t const>(
                                  (*edgelist_edge_types).data(), (*edgelist_edge_types).size()))
                              : cuda::std::nullopt,
          edgelist_edge_start_times
            ? cuda::std::make_optional(raft::device_span<edge_time_t const>(
                (*edgelist_edge_start_times).data(), (*edgelist_edge_start_times).size()))
            : cuda::std::nullopt,
          edgelist_edge_end_times
            ? cuda::std::make_optional(raft::device_span<edge_time_t const>(
                (*edgelist_edge_end_times).data(), (*edgelist_edge_end_times).size()))
            : cuda::std::nullopt},
        mem_frugal_threshold,
        keep_min_value_edge);

    if (edgelist_weights) {
      rmm::device_uvector<weight_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_weights->begin(),
                     tmp.begin());

      edgelist_weights = std::move(tmp);
    }

    if (edgelist_edge_ids) {
      rmm::device_uvector<edge_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_ids->begin(),
                     tmp.begin());

      edgelist_edge_ids = std::move(tmp);
    }

    if (edgelist_edge_types) {
      rmm::device_uvector<edge_type_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_types->begin(),
                     tmp.begin());

      edgelist_edge_types = std::move(tmp);
    }

    if (edgelist_edge_start_times) {
      rmm::device_uvector<edge_time_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_start_times->begin(),
                     tmp.begin());

      edgelist_edge_start_times = std::move(tmp);
    }

    if (edgelist_edge_end_times) {
      rmm::device_uvector<edge_time_t> tmp(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_end_times->begin(),
                     tmp.begin());

      edgelist_edge_end_times = std::move(tmp);
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
                         std::move(edgelist_edge_types),
                         std::move(edgelist_edge_start_times),
                         std::move(edgelist_edge_end_times));
}

}  // namespace cugraph

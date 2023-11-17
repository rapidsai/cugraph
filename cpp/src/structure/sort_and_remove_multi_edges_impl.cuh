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

#include <cugraph/utilities/dataframe_buffer.hpp>
// FIXME: mem_frugal_partition should probably not be in shuffle_comm.hpp
//        It's used here without any notion of shuffling
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuco/hash_functions.cuh>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <optional>

namespace cugraph {

namespace detail {

template <typename InputIterator, typename hash_op_t>
rmm::device_uvector<size_t> compute_hash_sizes(InputIterator iter_first,
                                               InputIterator iter_last,
                                               size_t num_hash_buckets,
                                               hash_op_t hash_op,
                                               rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<size_t> hash_counts(num_hash_buckets, stream_view);
  thrust::fill(rmm::exec_policy(stream_view), hash_counts.begin(), hash_counts.end(), size_t{0});

  auto hash_counts_view = raft::device_span<size_t>(hash_counts.data(), hash_counts.size());
  thrust::for_each(rmm::exec_policy(stream_view),
                   iter_first,
                   iter_last,
                   [num_hash_buckets, hash_op, hash_counts_view] __device__(auto value) {
                     atomicAdd(&hash_counts_view[hash_op(value) % num_hash_buckets], size_t{1});
                   });

  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), hash_counts.begin(), hash_counts.end(), hash_counts.begin());

  return hash_counts;
}

template <typename vertex_t>
struct hash_src_dst_pair {
  int32_t __device__ operator()(thrust::tuple<vertex_t, vertex_t> t) const
  {
    vertex_t pair[2];
    pair[0] = thrust::get<0>(t);
    pair[1] = thrust::get<1>(t);
    cuco::detail::MurmurHash3_32<vertex_t*> hash_func{};
    return hash_func.compute_hash(reinterpret_cast<std::byte*>(pair), 2 * sizeof(vertex_t));
  }
};

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
sort_and_remove_multi_edges(raft::handle_t const& handle,
                            rmm::device_uvector<vertex_t>&& edgelist_srcs,
                            rmm::device_uvector<vertex_t>&& edgelist_dsts,
                            size_t mem_frugal_threshold)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  if (edgelist_srcs.size() > mem_frugal_threshold) {
    // Tuning parameter to address high frequency multi-edges
    size_t num_hash_buckets{16};

    auto hash_counts = compute_hash_sizes(pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          num_hash_buckets,
                                          hash_src_dst_pair<vertex_t>{},
                                          handle.get_stream());

    auto pivot =
      static_cast<int32_t>(thrust::distance(hash_counts.begin(),
                                            thrust::lower_bound(handle.get_thrust_policy(),
                                                                hash_counts.begin(),
                                                                hash_counts.end(),
                                                                edgelist_srcs.size() / 2)));

    auto second_first = detail::mem_frugal_partition(pair_first,
                                                     pair_first + edgelist_srcs.size(),
                                                     hash_src_dst_pair<vertex_t>{},
                                                     pivot,
                                                     handle.get_stream());
    thrust::sort(handle.get_thrust_policy(), pair_first, second_first);
    thrust::sort(handle.get_thrust_policy(), second_first, pair_first + edgelist_srcs.size());
  } else {
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + edgelist_srcs.size());
  }

  edgelist_srcs.resize(
    thrust::distance(pair_first,
                     thrust::unique(handle.get_thrust_policy(),
                                    pair_first,
                                    pair_first + edgelist_srcs.size(),
                                    [] __device__(auto lhs, auto rhs) {
                                      return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                             (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                    })),
    handle.get_stream());

  edgelist_dsts.resize(edgelist_srcs.size(), handle.get_stream());

  return std::make_tuple(std::move(edgelist_srcs), std::move(edgelist_dsts));
}

template <typename vertex_t, typename edge_value_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           decltype(allocate_dataframe_buffer<edge_value_t>(size_t{0}, rmm::cuda_stream_view{}))>
sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  decltype(allocate_dataframe_buffer<edge_value_t>(0, rmm::cuda_stream_view{}))&& edgelist_values,
  size_t mem_frugal_threshold)
{
  auto pair_first = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

  if (edgelist_srcs.size() > mem_frugal_threshold) {
    // Tuning parameter to address high frequency multi-edges
    size_t num_hash_buckets{16};

    auto hash_counts = compute_hash_sizes(pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          num_hash_buckets,
                                          hash_src_dst_pair<vertex_t>{},
                                          handle.get_stream());

    auto pivot =
      static_cast<int32_t>(thrust::distance(hash_counts.begin(),
                                            thrust::lower_bound(handle.get_thrust_policy(),
                                                                hash_counts.begin(),
                                                                hash_counts.end(),
                                                                edgelist_srcs.size() / 2)));

    auto second_first = detail::mem_frugal_partition(pair_first,
                                                     pair_first + edgelist_srcs.size(),
                                                     get_dataframe_buffer_begin(edgelist_values),
                                                     hash_src_dst_pair<vertex_t>{},
                                                     pivot,
                                                     handle.get_stream());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        pair_first,
                        std::get<0>(second_first),
                        get_dataframe_buffer_begin(edgelist_values));
    thrust::sort_by_key(handle.get_thrust_policy(),
                        std::get<0>(second_first),
                        pair_first + edgelist_srcs.size(),
                        std::get<1>(second_first));
  } else {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        pair_first,
                        pair_first + edgelist_srcs.size(),
                        get_dataframe_buffer_begin(edgelist_values));
  }

  edgelist_srcs.resize(thrust::distance(pair_first,
                                        thrust::get<0>(thrust::unique_by_key(
                                          handle.get_thrust_policy(),
                                          pair_first,
                                          pair_first + edgelist_srcs.size(),
                                          get_dataframe_buffer_begin(edgelist_values),
                                          [] __device__(auto lhs, auto rhs) {
                                            return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                                   (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                          }))),
                       handle.get_stream());

  edgelist_dsts.resize(edgelist_srcs.size(), handle.get_stream());
  resize_dataframe_buffer(edgelist_values, edgelist_srcs.size(), handle.get_stream());

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
sort_and_remove_multi_edges(raft::handle_t const& handle,
                            rmm::device_uvector<vertex_t>&& edgelist_srcs,
                            rmm::device_uvector<vertex_t>&& edgelist_dsts,
                            std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                            std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                            std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types)
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
          detail::sort_and_remove_multi_edges<vertex_t,
                                              thrust::tuple<weight_t, edge_t, edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights),
                            std::move(*edgelist_edge_ids),
                            std::move(*edgelist_edge_types)),
            mem_frugal_threshold);
      } else {
        std::forward_as_tuple(
          edgelist_srcs, edgelist_dsts, std::tie(edgelist_weights, edgelist_edge_ids)) =
          detail::sort_and_remove_multi_edges<vertex_t, thrust::tuple<weight_t, edge_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights), std::move(*edgelist_edge_ids)),
            mem_frugal_threshold);
      }
    } else {
      if (edgelist_edge_types) {
        std::forward_as_tuple(
          edgelist_srcs, edgelist_dsts, std::tie(edgelist_weights, edgelist_edge_types)) =
          detail::sort_and_remove_multi_edges<vertex_t, thrust::tuple<weight_t, edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights), std::move(*edgelist_edge_types)),
            mem_frugal_threshold);
      } else {
        std::forward_as_tuple(edgelist_srcs, edgelist_dsts, std::tie(edgelist_weights)) =
          detail::sort_and_remove_multi_edges<vertex_t, thrust::tuple<weight_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights)),
            mem_frugal_threshold);
      }
    }
  } else {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        std::forward_as_tuple(
          edgelist_srcs, edgelist_dsts, std::tie(edgelist_edge_ids, edgelist_edge_types)) =
          detail::sort_and_remove_multi_edges<vertex_t, thrust::tuple<edge_t, edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_ids), std::move(*edgelist_edge_types)),
            mem_frugal_threshold);
      } else {
        std::forward_as_tuple(edgelist_srcs, edgelist_dsts, std::tie(edgelist_edge_ids)) =
          detail::sort_and_remove_multi_edges<vertex_t, thrust::tuple<edge_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_ids)),
            mem_frugal_threshold);
      }
    } else {
      if (edgelist_edge_types) {
        std::forward_as_tuple(edgelist_srcs, edgelist_dsts, std::tie(edgelist_edge_types)) =
          detail::sort_and_remove_multi_edges<vertex_t, thrust::tuple<edge_type_t>>(
            handle,
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_types)),
            mem_frugal_threshold);
      } else {
        std::tie(edgelist_srcs, edgelist_dsts) = detail::sort_and_remove_multi_edges(
          handle, std::move(edgelist_srcs), std::move(edgelist_dsts), mem_frugal_threshold);
      }
    }
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types));
}

}  // namespace cugraph

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

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

template <typename GroupIdIterator>
struct compute_group_id_count_pair_t {
  GroupIdIterator group_id_first{};
  GroupIdIterator group_id_last{};

  __device__ thrust::tuple<int, size_t> operator()(size_t i) const
  {
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<GroupIdIterator>::value_type, int>);
    auto lower_it =
      thrust::lower_bound(thrust::seq, group_id_first, group_id_last, static_cast<int>(i));
    auto upper_it = thrust::upper_bound(thrust::seq, lower_it, group_id_last, static_cast<int>(i));
    return thrust::make_tuple(static_cast<int>(i),
                              static_cast<size_t>(thrust::distance(lower_it, upper_it)));
  }
};

// inline to suppress a complaint about ODR violation
inline std::tuple<std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>,
                  std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>>
compute_tx_rx_counts_offsets_ranks(raft::comms::comms_t const& comm,
                                   rmm::device_uvector<size_t> const& d_tx_value_counts,
                                   rmm::cuda_stream_view stream_view)
{
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_rx_value_counts(comm_size, stream_view);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released.
  std::vector<size_t> tx_counts(comm_size, size_t{1});
  std::vector<size_t> tx_offsets(comm_size);
  std::iota(tx_offsets.begin(), tx_offsets.end(), size_t{0});
  std::vector<int> tx_dst_ranks(comm_size);
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::vector<size_t> rx_counts(comm_size, size_t{1});
  std::vector<size_t> rx_offsets(comm_size);
  std::iota(rx_offsets.begin(), rx_offsets.end(), size_t{0});
  std::vector<int> rx_src_ranks(comm_size);
  std::iota(rx_src_ranks.begin(), rx_src_ranks.end(), int{0});
  device_multicast_sendrecv(comm,
                            d_tx_value_counts.data(),
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            d_rx_value_counts.data(),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream_view);

  raft::update_host(tx_counts.data(), d_tx_value_counts.data(), comm_size, stream_view.value());
  raft::update_host(rx_counts.data(), d_rx_value_counts.data(), comm_size, stream_view.value());

  stream_view.synchronize();

  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_offsets.begin() + 1);
  std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, rx_offsets.begin() + 1);

  int num_tx_dst_ranks{0};
  int num_rx_src_ranks{0};
  for (int i = 0; i < comm_size; ++i) {
    if (tx_counts[i] != 0) {
      tx_counts[num_tx_dst_ranks]    = tx_counts[i];
      tx_offsets[num_tx_dst_ranks]   = tx_offsets[i];
      tx_dst_ranks[num_tx_dst_ranks] = tx_dst_ranks[i];
      ++num_tx_dst_ranks;
    }
    if (rx_counts[i] != 0) {
      rx_counts[num_rx_src_ranks]    = rx_counts[i];
      rx_offsets[num_rx_src_ranks]   = rx_offsets[i];
      rx_src_ranks[num_rx_src_ranks] = rx_src_ranks[i];
      ++num_rx_src_ranks;
    }
  }
  tx_counts.resize(num_tx_dst_ranks);
  tx_offsets.resize(num_tx_dst_ranks);
  tx_dst_ranks.resize(num_tx_dst_ranks);
  rx_counts.resize(num_rx_src_ranks);
  rx_offsets.resize(num_rx_src_ranks);
  rx_src_ranks.resize(num_rx_src_ranks);

  return std::make_tuple(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks);
}

template <typename key_type, typename KeyToGroupIdOp>
struct key_group_id_less_t {
  KeyToGroupIdOp key_to_group_id_op{};
  int pivot{};
  __device__ bool operator()(key_type k) const { return key_to_group_id_op(k) < pivot; }
};

template <typename value_type, typename ValueToGroupIdOp>
struct value_group_id_less_t {
  ValueToGroupIdOp value_to_group_id_op{};
  int pivot{};
  __device__ bool operator()(value_type v) const { return value_to_group_id_op(v) < pivot; }
};

template <typename key_type, typename value_type, typename KeyToGroupIdOp>
struct kv_pair_group_id_less_t {
  KeyToGroupIdOp key_to_group_id_op{};
  int pivot{};
  __device__ bool operator()(thrust::tuple<key_type, value_type> t) const
  {
    return key_to_group_id_op(thrust::get<0>(t)) < pivot;
  }
};

template <typename value_type, typename ValueToGroupIdOp>
struct value_group_id_greater_equal_t {
  ValueToGroupIdOp value_to_group_id_op{};
  int pivot{};
  __device__ bool operator()(value_type v) const { return value_to_group_id_op(v) >= pivot; }
};

template <typename key_type, typename value_type, typename KeyToGroupIdOp>
struct kv_pair_group_id_greater_equal_t {
  KeyToGroupIdOp key_to_group_id_op{};
  int pivot{};
  __device__ bool operator()(thrust::tuple<key_type, value_type> t) const
  {
    return key_to_group_id_op(thrust::get<0>(t)) >= pivot;
  }
};

template <typename ValueIterator>
void swap_partitions(ValueIterator value_first,
                     ValueIterator value_last,
                     size_t first_partition_size,
                     rmm::cuda_stream_view stream_view)
{
  auto num_elements          = static_cast<size_t>(thrust::distance(value_first, value_last));
  auto second_partition_size = num_elements - first_partition_size;
  if (first_partition_size >= second_partition_size) {
    auto tmp_value_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
        first_partition_size, stream_view);

    thrust::copy(rmm::exec_policy(stream_view),
                 value_first,
                 value_first + first_partition_size,
                 get_dataframe_buffer_begin(tmp_value_buffer));

    thrust::copy(rmm::exec_policy(stream_view),
                 value_first + first_partition_size,
                 value_first + num_elements,
                 value_first);

    thrust::copy(rmm::exec_policy(stream_view),
                 get_dataframe_buffer_begin(tmp_value_buffer),
                 get_dataframe_buffer_end(tmp_value_buffer),
                 value_first + second_partition_size);
  } else {
    auto tmp_value_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
        second_partition_size, stream_view);

    thrust::copy(rmm::exec_policy(stream_view),
                 value_first + first_partition_size,
                 value_first + num_elements,
                 get_dataframe_buffer_begin(tmp_value_buffer));

    thrust::copy(rmm::exec_policy(stream_view),
                 value_first,
                 value_first + first_partition_size,
                 value_first + (num_elements - first_partition_size));

    thrust::copy(rmm::exec_policy(stream_view),
                 get_dataframe_buffer_begin(tmp_value_buffer),
                 get_dataframe_buffer_end(tmp_value_buffer),
                 value_first);
  }
}

template <typename KeyIterator, typename ValueIterator>
void swap_partitions(KeyIterator key_first,
                     KeyIterator key_last,
                     ValueIterator value_first,
                     size_t first_partition_size,
                     rmm::cuda_stream_view stream_view)
{
  auto num_elements          = static_cast<size_t>(thrust::distance(key_first, key_last));
  auto second_partition_size = num_elements - first_partition_size;
  if (first_partition_size >= second_partition_size) {
    auto tmp_key_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<KeyIterator>::value_type>(
        first_partition_size, stream_view);
    auto tmp_value_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
        first_partition_size, stream_view);

    thrust::copy(rmm::exec_policy(stream_view),
                 key_first,
                 key_first + first_partition_size,
                 get_dataframe_buffer_begin(tmp_key_buffer));
    thrust::copy(rmm::exec_policy(stream_view),
                 value_first,
                 value_first + first_partition_size,
                 get_dataframe_buffer_begin(tmp_value_buffer));

    thrust::copy(rmm::exec_policy(stream_view),
                 key_first + first_partition_size,
                 key_first + num_elements,
                 key_first);
    thrust::copy(rmm::exec_policy(stream_view),
                 value_first + first_partition_size,
                 value_first + num_elements,
                 value_first);

    thrust::copy(rmm::exec_policy(stream_view),
                 get_dataframe_buffer_begin(tmp_key_buffer),
                 get_dataframe_buffer_end(tmp_key_buffer),
                 key_first + second_partition_size);
    thrust::copy(rmm::exec_policy(stream_view),
                 get_dataframe_buffer_begin(tmp_value_buffer),
                 get_dataframe_buffer_end(tmp_value_buffer),
                 value_first + second_partition_size);
  } else {
    auto tmp_key_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<KeyIterator>::value_type>(
        second_partition_size, stream_view);
    auto tmp_value_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
        second_partition_size, stream_view);

    thrust::copy(rmm::exec_policy(stream_view),
                 key_first + first_partition_size,
                 key_first + num_elements,
                 get_dataframe_buffer_begin(tmp_key_buffer));
    thrust::copy(rmm::exec_policy(stream_view),
                 value_first + first_partition_size,
                 value_first + num_elements,
                 get_dataframe_buffer_begin(tmp_value_buffer));

    thrust::copy(rmm::exec_policy(stream_view),
                 key_first,
                 key_first + first_partition_size,
                 key_first + (num_elements - first_partition_size));
    thrust::copy(rmm::exec_policy(stream_view),
                 value_first,
                 value_first + first_partition_size,
                 value_first + (num_elements - first_partition_size));

    thrust::copy(rmm::exec_policy(stream_view),
                 get_dataframe_buffer_begin(tmp_key_buffer),
                 get_dataframe_buffer_end(tmp_key_buffer),
                 key_first);
    thrust::copy(rmm::exec_policy(stream_view),
                 get_dataframe_buffer_begin(tmp_value_buffer),
                 get_dataframe_buffer_end(tmp_value_buffer),
                 value_first);
  }
}

// Use roughly half temporary buffer than thrust::partition (if first & second partition sizes are
// comparable). This also uses multiple smaller allocations than one single allocation (thrust::sort
// does this) of the same aggregate size if the input iterators are the zip iterators (this is more
// favorable to the pool allocator).
template <typename ValueIterator, typename ValueToGroupIdOp>
ValueIterator mem_frugal_partition(
  ValueIterator value_first,
  ValueIterator value_last,
  ValueToGroupIdOp value_to_group_id_op,
  int pivot,  // group id less than pivot goes to the first partition
  rmm::cuda_stream_view stream_view)
{
  auto num_elements = static_cast<size_t>(thrust::distance(value_first, value_last));
  auto first_size   = static_cast<size_t>(thrust::count_if(
    rmm::exec_policy(stream_view),
    value_first,
    value_last,
    value_group_id_less_t<typename thrust::iterator_traits<ValueIterator>::value_type,
                          ValueToGroupIdOp>{value_to_group_id_op, pivot}));
  auto second_size  = num_elements - first_size;

  auto tmp_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      second_size, stream_view);

  // to limit memory footprint (16 * 1024 * 1024 is a tuning parameter)
  // thrust::copy_if (1.15.0) also uses temporary buffer
  auto constexpr max_elements_per_iteration = size_t{16} * 1024 * 1024;
  auto num_chunks = (num_elements + max_elements_per_iteration - 1) / max_elements_per_iteration;
  auto output_chunk_first = get_dataframe_buffer_begin(tmp_buffer);
  for (size_t i = 0; i < num_chunks; ++i) {
    output_chunk_first = thrust::copy_if(
      rmm::exec_policy(stream_view),
      value_first + max_elements_per_iteration * i,
      value_first + std::min(max_elements_per_iteration * (i + 1), num_elements),
      output_chunk_first,
      value_group_id_greater_equal_t<typename thrust::iterator_traits<ValueIterator>::value_type,
                                     ValueToGroupIdOp>{value_to_group_id_op, pivot});
  }

  thrust::remove_if(
    rmm::exec_policy(stream_view),
    value_first,
    value_last,
    value_group_id_greater_equal_t<typename thrust::iterator_traits<ValueIterator>::value_type,
                                   ValueToGroupIdOp>{value_to_group_id_op, pivot});
  thrust::copy(rmm::exec_policy(stream_view),
               get_dataframe_buffer_cbegin(tmp_buffer),
               get_dataframe_buffer_cend(tmp_buffer),
               value_first + first_size);

  return value_first + first_size;
}

// Use roughly half temporary buffer than thrust::partition (if first & second partition sizes are
// comparable). This also uses multiple smaller allocations than one single allocation (thrust::sort
// does this) of the same aggregate size if the input iterators are the zip iterators (this is more
// favorable to the pool allocator).
template <typename KeyIterator, typename ValueIterator, typename KeyToGroupIdOp>
std::tuple<KeyIterator, ValueIterator> mem_frugal_partition(
  KeyIterator key_first,
  KeyIterator key_last,
  ValueIterator value_first,
  KeyToGroupIdOp key_to_group_id_op,
  int pivot,  // group Id less than pivot goes to the first partition
  rmm::cuda_stream_view stream_view)
{
  auto num_elements = static_cast<size_t>(thrust::distance(key_first, key_last));
  auto first_size   = static_cast<size_t>(thrust::count_if(
    rmm::exec_policy(stream_view),
    key_first,
    key_last,
    key_group_id_less_t<typename thrust::iterator_traits<KeyIterator>::value_type, KeyToGroupIdOp>{
      key_to_group_id_op, pivot}));
  auto second_size  = num_elements - first_size;

  auto tmp_key_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<KeyIterator>::value_type>(
      second_size, stream_view);
  auto tmp_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      second_size, stream_view);

  // to limit memory footprint (16 * 1024 * 1024 is a tuning parameter)
  // thrust::copy_if (1.15.0) also uses temporary buffer
  auto max_elements_per_iteration = size_t{16} * 1024 * 1024;
  auto num_chunks    = (num_elements + max_elements_per_iteration - 1) / max_elements_per_iteration;
  auto kv_pair_first = thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first));
  auto output_chunk_first = thrust::make_zip_iterator(thrust::make_tuple(
    get_dataframe_buffer_begin(tmp_key_buffer), get_dataframe_buffer_begin(tmp_value_buffer)));
  for (size_t i = 0; i < num_chunks; ++i) {
    output_chunk_first = thrust::copy_if(
      rmm::exec_policy(stream_view),
      kv_pair_first + max_elements_per_iteration * i,
      kv_pair_first + std::min(max_elements_per_iteration * (i + 1), num_elements),
      output_chunk_first,
      kv_pair_group_id_greater_equal_t<typename thrust::iterator_traits<KeyIterator>::value_type,
                                       typename thrust::iterator_traits<ValueIterator>::value_type,
                                       KeyToGroupIdOp>{key_to_group_id_op, pivot});
  }

  thrust::remove_if(
    rmm::exec_policy(stream_view),
    kv_pair_first,
    kv_pair_first + num_elements,
    kv_pair_group_id_greater_equal_t<typename thrust::iterator_traits<KeyIterator>::value_type,
                                     typename thrust::iterator_traits<ValueIterator>::value_type,
                                     KeyToGroupIdOp>{key_to_group_id_op, pivot});
  thrust::copy(rmm::exec_policy(stream_view),
               get_dataframe_buffer_cbegin(tmp_key_buffer),
               get_dataframe_buffer_cend(tmp_key_buffer),
               key_first + first_size);
  thrust::copy(rmm::exec_policy(stream_view),
               get_dataframe_buffer_cbegin(tmp_value_buffer),
               get_dataframe_buffer_cend(tmp_value_buffer),
               value_first + first_size);

  return std::make_tuple(key_first + first_size, value_first + first_size);
}

template <typename ValueIterator, typename ValueToGroupIdOp>
void mem_frugal_groupby(
  ValueIterator value_first,
  ValueIterator value_last,
  ValueToGroupIdOp value_to_group_id_op,
  int num_groups,
  size_t mem_frugal_threshold,  // take the memory frugal approach (instead of thrust::sort) if #
                                // elements to groupby is no smaller than this value
  rmm::cuda_stream_view stream_view)
{
  std::vector<int> group_firsts{};
  std::vector<int> group_lasts{};
  std::vector<ValueIterator> value_firsts{};
  std::vector<ValueIterator> value_lasts{};
  if (num_groups > 1) {
    group_firsts.push_back(int{0});
    group_lasts.push_back(num_groups);
    value_firsts.push_back(value_first);
    value_lasts.push_back(value_last);
  }

  auto offset_first = size_t{0};
  auto offset_last  = group_firsts.size();
  while (offset_first < offset_last) {
    for (size_t i = offset_first; i < offset_last; ++i) {
      auto pivot = (group_firsts[i] + group_lasts[i]) / 2;
      if (static_cast<size_t>(thrust::distance(value_firsts[i], value_lasts[i])) <
          mem_frugal_threshold) {
        if (group_lasts[i] - group_firsts[i] == 2) {
          thrust::partition(
            rmm::exec_policy(stream_view),
            value_firsts[i],
            value_lasts[i],
            value_group_id_less_t<typename thrust::iterator_traits<ValueIterator>::value_type,
                                  ValueToGroupIdOp>{value_to_group_id_op, pivot});
        } else {
          thrust::sort(rmm::exec_policy(stream_view),
                       value_firsts[i],
                       value_lasts[i],
                       [value_to_group_id_op] __device__(auto lhs, auto rhs) {
                         return value_to_group_id_op(lhs) < value_to_group_id_op(rhs);
                       });
        }
      } else {
        ValueIterator second_first{};
        auto num_elements = static_cast<size_t>(thrust::distance(value_firsts[i], value_lasts[i]));
        auto first_chunk_partition_first  = mem_frugal_partition(value_firsts[i],
                                                                value_firsts[i] + num_elements / 2,
                                                                value_to_group_id_op,
                                                                pivot,
                                                                stream_view);
        auto second_chunk_partition_first = mem_frugal_partition(value_firsts[i] + num_elements / 2,
                                                                 value_lasts[i],
                                                                 value_to_group_id_op,
                                                                 pivot,
                                                                 stream_view);
        auto no_less_size                 = static_cast<size_t>(
          thrust::distance(first_chunk_partition_first, value_firsts[i] + num_elements / 2));
        auto less_size = static_cast<size_t>(
          thrust::distance(value_firsts[i] + num_elements / 2, second_chunk_partition_first));
        swap_partitions(value_firsts[i] + (num_elements / 2 - no_less_size),
                        value_firsts[i] + (num_elements / 2 + less_size),
                        no_less_size,
                        stream_view);

        second_first = value_firsts[i] + ((num_elements / 2 - no_less_size) + less_size);
        if (pivot - group_firsts[i] > 1) {
          group_firsts.push_back(group_firsts[i]);
          group_lasts.push_back(pivot);
          value_firsts.push_back(value_firsts[i]);
          value_lasts.push_back(second_first);
        }
        if (group_lasts[i] - pivot > 1) {
          group_firsts.push_back(pivot);
          group_lasts.push_back(group_lasts[i]);
          value_firsts.push_back(second_first);
          value_lasts.push_back(value_lasts[i]);
        }
      }
    }
    offset_first = offset_last;
    offset_last  = group_firsts.size();
  }
}

template <typename KeyIterator, typename ValueIterator, typename KeyToGroupIdOp>
void mem_frugal_groupby(
  KeyIterator key_first,
  KeyIterator key_last,
  ValueIterator value_first,
  KeyToGroupIdOp key_to_group_id_op,
  int num_groups,
  size_t mem_frugal_threshold,  // take the memory frugal approach (instead of thrust::sort) if #
                                // elements to groupby is no smaller than this value
  rmm::cuda_stream_view stream_view)
{
  std::vector<int> group_firsts{};
  std::vector<int> group_lasts{};
  std::vector<KeyIterator> key_firsts{};
  std::vector<KeyIterator> key_lasts{};
  std::vector<ValueIterator> value_firsts{};
  if (num_groups > 1) {
    group_firsts.push_back(int{0});
    group_lasts.push_back(num_groups);
    key_firsts.push_back(key_first);
    key_lasts.push_back(key_last);
    value_firsts.push_back(value_first);
  }

  auto offset_first = size_t{0};
  auto offset_last  = group_firsts.size();
  while (offset_first < offset_last) {
    for (size_t i = offset_first; i < offset_last; ++i) {
      auto pivot = (group_firsts[i] + group_lasts[i]) / 2;
      if (static_cast<size_t>(thrust::distance(key_firsts[i], key_lasts[i])) <
          mem_frugal_threshold) {
        if (group_lasts[i] - group_firsts[i] == 2) {
          auto kv_pair_first =
            thrust::make_zip_iterator(thrust::make_tuple(key_firsts[i], value_firsts[i]));
          thrust::partition(
            rmm::exec_policy(stream_view),
            kv_pair_first,
            kv_pair_first + thrust::distance(key_firsts[i], key_lasts[i]),
            kv_pair_group_id_less_t<typename thrust::iterator_traits<KeyIterator>::value_type,
                                    typename thrust::iterator_traits<ValueIterator>::value_type,
                                    KeyToGroupIdOp>{key_to_group_id_op, pivot});
        } else {
          thrust::sort_by_key(rmm::exec_policy(stream_view),
                              key_firsts[i],
                              key_lasts[i],
                              value_firsts[i],
                              [key_to_group_id_op] __device__(auto lhs, auto rhs) {
                                return key_to_group_id_op(lhs) < key_to_group_id_op(rhs);
                              });
        }
      } else {
        std::tuple<KeyIterator, ValueIterator> second_first{};
        auto num_elements = static_cast<size_t>(thrust::distance(key_firsts[i], key_lasts[i]));
        auto first_chunk_partition_first  = mem_frugal_partition(key_firsts[i],
                                                                key_firsts[i] + num_elements / 2,
                                                                value_firsts[i],
                                                                key_to_group_id_op,
                                                                pivot,
                                                                stream_view);
        auto second_chunk_partition_first = mem_frugal_partition(key_firsts[i] + num_elements / 2,
                                                                 key_lasts[i],
                                                                 value_firsts[i] + num_elements / 2,
                                                                 key_to_group_id_op,
                                                                 pivot,
                                                                 stream_view);
        auto no_less_size                 = static_cast<size_t>(thrust::distance(
          std::get<0>(first_chunk_partition_first), key_firsts[i] + num_elements / 2));
        auto less_size                    = static_cast<size_t>(thrust::distance(
          key_firsts[i] + num_elements / 2, std::get<0>(second_chunk_partition_first)));
        swap_partitions(key_firsts[i] + (num_elements / 2 - no_less_size),
                        key_firsts[i] + (num_elements / 2 + less_size),
                        value_firsts[i] + (num_elements / 2 - no_less_size),
                        no_less_size,
                        stream_view);

        second_first =
          std::make_tuple(key_firsts[i] + ((num_elements / 2 - no_less_size) + less_size),
                          value_firsts[i] + ((num_elements / 2 - no_less_size) + less_size));
        if (pivot - group_firsts[i] > 1) {
          group_firsts.push_back(group_firsts[i]);
          group_lasts.push_back(pivot);
          key_firsts.push_back(key_firsts[i]);
          key_lasts.push_back(std::get<0>(second_first));
          value_firsts.push_back(value_firsts[i]);
        }
        if (group_lasts[i] - pivot > 1) {
          group_firsts.push_back(pivot);
          group_lasts.push_back(group_lasts[i]);
          key_firsts.push_back(std::get<0>(second_first));
          key_lasts.push_back(key_lasts[i]);
          value_firsts.push_back(std::get<1>(second_first));
        }
      }
    }
    offset_first = offset_last;
    offset_last  = group_firsts.size();
  }
}

}  // namespace detail

template <typename ValueIterator, typename ValueToGroupIdOp>
rmm::device_uvector<size_t> groupby_and_count(ValueIterator tx_value_first /* [INOUT */,
                                              ValueIterator tx_value_last /* [INOUT */,
                                              ValueToGroupIdOp value_to_group_id_op,
                                              int num_groups,
                                              size_t mem_frugal_threshold,
                                              rmm::cuda_stream_view stream_view)
{
  detail::mem_frugal_groupby(tx_value_first,
                             tx_value_last,
                             value_to_group_id_op,
                             num_groups,
                             mem_frugal_threshold,
                             stream_view);

  auto group_id_first = thrust::make_transform_iterator(
    tx_value_first,
    [value_to_group_id_op] __device__(auto value) { return value_to_group_id_op(value); });
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream_view);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream_view);
  auto rank_count_pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_tx_dst_ranks.begin(), d_tx_value_counts.begin()));
  thrust::tabulate(
    rmm::exec_policy(stream_view),
    rank_count_pair_first,
    rank_count_pair_first + num_groups,
    detail::compute_group_id_count_pair_t<decltype(group_id_first)>{
      group_id_first, group_id_first + thrust::distance(tx_value_first, tx_value_last)});

  return d_tx_value_counts;
}

template <typename VertexIterator, typename ValueIterator, typename KeyToGroupIdOp>
rmm::device_uvector<size_t> groupby_and_count(VertexIterator tx_key_first /* [INOUT */,
                                              VertexIterator tx_key_last /* [INOUT */,
                                              ValueIterator tx_value_first /* [INOUT */,
                                              KeyToGroupIdOp key_to_group_id_op,
                                              int num_groups,
                                              size_t mem_frugal_threshold,
                                              rmm::cuda_stream_view stream_view)
{
  detail::mem_frugal_groupby(tx_key_first,
                             tx_key_last,
                             tx_value_first,
                             key_to_group_id_op,
                             num_groups,
                             mem_frugal_threshold,
                             stream_view);

  auto group_id_first = thrust::make_transform_iterator(
    tx_key_first, [key_to_group_id_op] __device__(auto key) { return key_to_group_id_op(key); });
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream_view);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream_view);
  auto rank_count_pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_tx_dst_ranks.begin(), d_tx_value_counts.begin()));
  thrust::tabulate(rmm::exec_policy(stream_view),
                   rank_count_pair_first,
                   rank_count_pair_first + num_groups,
                   detail::compute_group_id_count_pair_t<decltype(group_id_first)>{
                     group_id_first, group_id_first + thrust::distance(tx_key_first, tx_key_last)});

  return d_tx_value_counts;
}

template <typename TxValueIterator>
auto shuffle_values(raft::comms::comms_t const& comm,
                    TxValueIterator tx_value_first,
                    std::vector<size_t> const& tx_value_counts,
                    rmm::cuda_stream_view stream_view)
{
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_tx_value_counts(comm_size, stream_view);
  raft::update_device(
    d_tx_value_counts.data(), tx_value_counts.data(), comm_size, stream_view.value());

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_offsets{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_offsets{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks) =
    detail::compute_tx_rx_counts_offsets_ranks(comm, d_tx_value_counts, stream_view);

  auto rx_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<TxValueIterator>::value_type>(
      rx_offsets.size() > 0 ? rx_offsets.back() + rx_counts.back() : size_t{0}, stream_view);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            get_dataframe_buffer_begin(rx_value_buffer),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream_view);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      assert(rx_src_ranks[i] < comm_size);
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_value_buffer), rx_counts);
}

template <typename ValueIterator, typename ValueToGPUIdOp>
auto groupby_gpu_id_and_shuffle_values(raft::comms::comms_t const& comm,
                                       ValueIterator tx_value_first /* [INOUT */,
                                       ValueIterator tx_value_last /* [INOUT */,
                                       ValueToGPUIdOp value_to_gpu_id_op,
                                       rmm::cuda_stream_view stream_view)
{
  auto const comm_size = comm.get_size();

  auto d_tx_value_counts = groupby_and_count(tx_value_first,
                                             tx_value_last,
                                             value_to_gpu_id_op,
                                             comm.get_size(),
                                             std::numeric_limits<size_t>::max(),
                                             stream_view);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_offsets{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_offsets{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks) =
    detail::compute_tx_rx_counts_offsets_ranks(comm, d_tx_value_counts, stream_view);

  auto rx_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      rx_offsets.size() > 0 ? rx_offsets.back() + rx_counts.back() : size_t{0}, stream_view);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            get_dataframe_buffer_begin(rx_value_buffer),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream_view);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_value_buffer), rx_counts);
}

template <typename VertexIterator, typename ValueIterator, typename KeyToGPUIdOp>
auto groupby_gpu_id_and_shuffle_kv_pairs(raft::comms::comms_t const& comm,
                                         VertexIterator tx_key_first /* [INOUT */,
                                         VertexIterator tx_key_last /* [INOUT */,
                                         ValueIterator tx_value_first /* [INOUT */,
                                         KeyToGPUIdOp key_to_gpu_id_op,
                                         rmm::cuda_stream_view stream_view)
{
  auto const comm_size = comm.get_size();

  auto d_tx_value_counts = groupby_and_count(tx_key_first,
                                             tx_key_last,
                                             tx_value_first,
                                             key_to_gpu_id_op,
                                             comm.get_size(),
                                             std::numeric_limits<size_t>::max(),
                                             stream_view);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_offsets{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_offsets{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks) =
    detail::compute_tx_rx_counts_offsets_ranks(comm, d_tx_value_counts, stream_view);

  rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type> rx_keys(
    rx_offsets.size() > 0 ? rx_offsets.back() + rx_counts.back() : size_t{0}, stream_view);
  auto rx_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      rx_keys.size(), stream_view);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_key_first,
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            rx_keys.begin(),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream_view);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            get_dataframe_buffer_begin(rx_value_buffer),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream_view);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      assert(rx_src_ranks[i] < comm_size);
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_keys), std::move(rx_value_buffer), rx_counts);
}

}  // namespace cugraph

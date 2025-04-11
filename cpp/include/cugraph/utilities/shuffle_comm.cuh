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

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
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
#include <thrust/unique.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

constexpr size_t cache_line_size = 128;

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
                              static_cast<size_t>(cuda::std::distance(lower_it, upper_it)));
  }
};

// inline to suppress a complaint about ODR violation
inline std::tuple<std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>,
                  std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>>
compute_tx_rx_counts_displs_ranks(raft::comms::comms_t const& comm,
                                  rmm::device_uvector<size_t> const& d_tx_value_counts,
                                  bool drop_empty_ranks,
                                  rmm::cuda_stream_view stream_view)
{
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_rx_value_counts(comm_size, stream_view);

  std::vector<size_t> tx_counts(comm_size, size_t{1});
  std::vector<size_t> tx_displs(comm_size);
  std::iota(tx_displs.begin(), tx_displs.end(), size_t{0});
  std::vector<int> tx_dst_ranks(comm_size);
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::vector<size_t> rx_counts(comm_size, size_t{1});
  std::vector<size_t> rx_displs(comm_size);
  std::iota(rx_displs.begin(), rx_displs.end(), size_t{0});
  std::vector<int> rx_src_ranks(comm_size);
  std::iota(rx_src_ranks.begin(), rx_src_ranks.end(), int{0});
  device_multicast_sendrecv(comm,
                            d_tx_value_counts.data(),
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            d_rx_value_counts.data(),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            stream_view);

  raft::update_host(tx_counts.data(), d_tx_value_counts.data(), comm_size, stream_view.value());
  raft::update_host(rx_counts.data(), d_rx_value_counts.data(), comm_size, stream_view.value());

  stream_view.synchronize();

  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_displs.begin() + 1);
  std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, rx_displs.begin() + 1);

  if (drop_empty_ranks) {
    int num_tx_dst_ranks{0};
    int num_rx_src_ranks{0};
    for (int i = 0; i < comm_size; ++i) {
      if (tx_counts[i] != 0) {
        tx_counts[num_tx_dst_ranks]    = tx_counts[i];
        tx_displs[num_tx_dst_ranks]    = tx_displs[i];
        tx_dst_ranks[num_tx_dst_ranks] = tx_dst_ranks[i];
        ++num_tx_dst_ranks;
      }
      if (rx_counts[i] != 0) {
        rx_counts[num_rx_src_ranks]    = rx_counts[i];
        rx_displs[num_rx_src_ranks]    = rx_displs[i];
        rx_src_ranks[num_rx_src_ranks] = rx_src_ranks[i];
        ++num_rx_src_ranks;
      }
    }
    tx_counts.resize(num_tx_dst_ranks);
    tx_displs.resize(num_tx_dst_ranks);
    tx_dst_ranks.resize(num_tx_dst_ranks);
    rx_counts.resize(num_rx_src_ranks);
    rx_displs.resize(num_rx_src_ranks);
    rx_src_ranks.resize(num_rx_src_ranks);
  }

  return std::make_tuple(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks);
}

template <typename key_type, typename KeyToGroupIdOp>
struct key_group_id_less_t {
  KeyToGroupIdOp key_to_group_id_op;
  int pivot{};
  __device__ bool operator()(key_type k) const { return key_to_group_id_op(k) < pivot; }
};

template <typename value_type, typename ValueToGroupIdOp>
struct value_group_id_less_t {
  ValueToGroupIdOp value_to_group_id_op;
  int pivot{};
  __device__ bool operator()(value_type v) const { return value_to_group_id_op(v) < pivot; }
};

template <typename key_type, typename value_type, typename KeyToGroupIdOp>
struct kv_pair_group_id_less_t {
  KeyToGroupIdOp key_to_group_id_op;
  int pivot{};
  __device__ bool operator()(thrust::tuple<key_type, value_type> t) const
  {
    return key_to_group_id_op(thrust::get<0>(t)) < pivot;
  }
};

template <typename value_type, typename ValueToGroupIdOp>
struct value_group_id_greater_equal_t {
  ValueToGroupIdOp value_to_group_id_op;
  int pivot{};
  __device__ bool operator()(value_type v) const { return value_to_group_id_op(v) >= pivot; }
};

template <typename key_type, typename value_type, typename KeyToGroupIdOp>
struct kv_pair_group_id_greater_equal_t {
  KeyToGroupIdOp key_to_group_id_op;
  int pivot{};
  __device__ bool operator()(thrust::tuple<key_type, value_type> t) const
  {
    return key_to_group_id_op(thrust::get<0>(t)) >= pivot;
  }
};

template <typename ValueIterator, typename ValueToGroupIdOp>
void multi_partition(ValueIterator value_first,
                     ValueIterator value_last,
                     ValueToGroupIdOp value_to_group_id_op,
                     int group_first,
                     int group_last,
                     rmm::cuda_stream_view stream_view)
{
  auto num_values = static_cast<size_t>(cuda::std::distance(value_first, value_last));
  auto num_groups = group_last - group_first;

  rmm::device_uvector<size_t> counts(num_groups, stream_view);
  rmm::device_uvector<int> group_ids(num_values, stream_view);
  rmm::device_uvector<size_t> intra_partition_displs(num_values, stream_view);
  thrust::fill(rmm::exec_policy(stream_view), counts.begin(), counts.end(), size_t{0});
  thrust::transform(
    rmm::exec_policy(stream_view),
    value_first,
    value_last,
    thrust::make_zip_iterator(
      thrust::make_tuple(group_ids.begin(), intra_partition_displs.begin())),
    cuda::proclaim_return_type<thrust::tuple<int, size_t>>(
      [value_to_group_id_op, group_first, counts = counts.data()] __device__(auto value) {
        auto group_id = value_to_group_id_op(value);
        cuda::std::atomic_ref<size_t> counter(counts[group_id - group_first]);
        return thrust::make_tuple(group_id,
                                  counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed));
      }));

  rmm::device_uvector<size_t> displacements(num_groups, stream_view);
  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), counts.begin(), counts.end(), displacements.begin());

  auto tmp_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      num_values, stream_view);
  auto input_triplet_first = thrust::make_zip_iterator(
    thrust::make_tuple(value_first, group_ids.begin(), intra_partition_displs.begin()));
  auto tmp_value_first = get_dataframe_buffer_begin(tmp_value_buffer);
  thrust::for_each(
    rmm::exec_policy(stream_view),
    input_triplet_first,
    input_triplet_first + num_values,
    [group_first,
     displacements = displacements.data(),
     output_first  = get_dataframe_buffer_begin(tmp_value_buffer)] __device__(auto triplet) {
      auto group_id            = thrust::get<1>(triplet);
      auto offset              = displacements[group_id - group_first] + thrust::get<2>(triplet);
      *(output_first + offset) = thrust::get<0>(triplet);
    });
  thrust::copy(
    rmm::exec_policy(stream_view), tmp_value_first, tmp_value_first + num_values, value_first);
}

template <typename KeyIterator, typename ValueIterator, typename KeyToGroupIdOp>
void multi_partition(KeyIterator key_first,
                     KeyIterator key_last,
                     ValueIterator value_first,
                     KeyToGroupIdOp key_to_group_id_op,
                     int group_first,
                     int group_last,
                     rmm::cuda_stream_view stream_view)
{
  auto num_keys   = static_cast<size_t>(cuda::std::distance(key_first, key_last));
  auto num_groups = group_last - group_first;

  rmm::device_uvector<size_t> counts(num_groups, stream_view);
  rmm::device_uvector<int> group_ids(num_keys, stream_view);
  rmm::device_uvector<size_t> intra_partition_displs(num_keys, stream_view);
  thrust::fill(rmm::exec_policy(stream_view), counts.begin(), counts.end(), size_t{0});
  thrust::transform(
    rmm::exec_policy(stream_view),
    key_first,
    key_last,
    thrust::make_zip_iterator(
      thrust::make_tuple(group_ids.begin(), intra_partition_displs.begin())),
    cuda::proclaim_return_type<thrust::tuple<int, size_t>>(
      [key_to_group_id_op, group_first, counts = counts.data()] __device__(auto key) {
        auto group_id = key_to_group_id_op(key);
        cuda::std::atomic_ref<size_t> counter(counts[group_id - group_first]);
        return thrust::make_tuple(group_id,
                                  counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed));
      }));

  rmm::device_uvector<size_t> displacements(num_groups, stream_view);
  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), counts.begin(), counts.end(), displacements.begin());

  auto tmp_key_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<KeyIterator>::value_type>(
      num_keys, stream_view);
  auto tmp_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      num_keys, stream_view);
  auto input_quadraplet_first = thrust::make_zip_iterator(
    thrust::make_tuple(key_first, value_first, group_ids.begin(), intra_partition_displs.begin()));
  auto tmp_kv_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
    get_dataframe_buffer_begin(tmp_key_buffer), get_dataframe_buffer_begin(tmp_value_buffer)));
  thrust::for_each(rmm::exec_policy(stream_view),
                   input_quadraplet_first,
                   input_quadraplet_first + num_keys,
                   [group_first,
                    displacements = displacements.data(),
                    output_first  = tmp_kv_pair_first] __device__(auto quadraplet) {
                     auto group_id = thrust::get<2>(quadraplet);
                     auto offset =
                       displacements[group_id - group_first] + thrust::get<3>(quadraplet);
                     *(output_first + offset) =
                       thrust::make_tuple(thrust::get<0>(quadraplet), thrust::get<1>(quadraplet));
                   });
  thrust::copy(rmm::exec_policy(stream_view),
               tmp_kv_pair_first,
               tmp_kv_pair_first + num_keys,
               thrust::make_zip_iterator(thrust::make_tuple(key_first, value_first)));
}

template <typename ValueIterator>
void swap_partitions(ValueIterator value_first,
                     ValueIterator value_last,
                     size_t first_partition_size,
                     rmm::cuda_stream_view stream_view)
{
  auto num_elements          = static_cast<size_t>(cuda::std::distance(value_first, value_last));
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
  auto num_elements          = static_cast<size_t>(cuda::std::distance(key_first, key_last));
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
  auto num_elements = static_cast<size_t>(cuda::std::distance(value_first, value_last));
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
  auto num_elements = static_cast<size_t>(cuda::std::distance(key_first, key_last));
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
      if (static_cast<size_t>(cuda::std::distance(value_firsts[i], value_lasts[i])) <
          mem_frugal_threshold) {
        if (group_lasts[i] - group_firsts[i] == 2) {
          thrust::partition(
            rmm::exec_policy(stream_view),
            value_firsts[i],
            value_lasts[i],
            value_group_id_less_t<typename thrust::iterator_traits<ValueIterator>::value_type,
                                  ValueToGroupIdOp>{value_to_group_id_op, pivot});
        } else {
#if 1  // FIXME: keep the both if and else cases till the performance improvement gets fully
       // validated. The else path should be eventually deleted.
          multi_partition(value_firsts[i],
                          value_lasts[i],
                          value_to_group_id_op,
                          group_firsts[i],
                          group_lasts[i],
                          stream_view);
#else
          thrust::sort(rmm::exec_policy(stream_view),
                       value_firsts[i],
                       value_lasts[i],
                       [value_to_group_id_op] __device__(auto lhs, auto rhs) {
                         return value_to_group_id_op(lhs) < value_to_group_id_op(rhs);
                       });
#endif
        }
      } else {
        ValueIterator second_first{};
        auto num_elements =
          static_cast<size_t>(cuda::std::distance(value_firsts[i], value_lasts[i]));
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
          cuda::std::distance(first_chunk_partition_first, value_firsts[i] + num_elements / 2));
        auto less_size = static_cast<size_t>(
          cuda::std::distance(value_firsts[i] + num_elements / 2, second_chunk_partition_first));
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
      if (static_cast<size_t>(cuda::std::distance(key_firsts[i], key_lasts[i])) <
          mem_frugal_threshold) {
        if (group_lasts[i] - group_firsts[i] == 2) {
          auto kv_pair_first =
            thrust::make_zip_iterator(thrust::make_tuple(key_firsts[i], value_firsts[i]));
          thrust::partition(
            rmm::exec_policy(stream_view),
            kv_pair_first,
            kv_pair_first + cuda::std::distance(key_firsts[i], key_lasts[i]),
            kv_pair_group_id_less_t<typename thrust::iterator_traits<KeyIterator>::value_type,
                                    typename thrust::iterator_traits<ValueIterator>::value_type,
                                    KeyToGroupIdOp>{key_to_group_id_op, pivot});
        } else {
#if 1  // FIXME: keep the both if and else cases till the performance improvement gets fully
       // validated. The else path should be eventually deleted.
          multi_partition(key_firsts[i],
                          key_lasts[i],
                          value_firsts[i],
                          key_to_group_id_op,
                          group_firsts[i],
                          group_lasts[i],
                          stream_view);
#else
          thrust::sort_by_key(rmm::exec_policy(stream_view),
                              key_firsts[i],
                              key_lasts[i],
                              value_firsts[i],
                              [key_to_group_id_op] __device__(auto lhs, auto rhs) {
                                return key_to_group_id_op(lhs) < key_to_group_id_op(rhs);
                              });
#endif
        }
      } else {
        std::tuple<KeyIterator, ValueIterator> second_first{};
        auto num_elements = static_cast<size_t>(cuda::std::distance(key_firsts[i], key_lasts[i]));
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
        auto no_less_size                 = static_cast<size_t>(cuda::std::distance(
          std::get<0>(first_chunk_partition_first), key_firsts[i] + num_elements / 2));
        auto less_size                    = static_cast<size_t>(cuda::std::distance(
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
    tx_value_first, cuda::proclaim_return_type<int>([value_to_group_id_op] __device__(auto value) {
      return value_to_group_id_op(value);
    }));
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream_view);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream_view);
  auto rank_count_pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_tx_dst_ranks.begin(), d_tx_value_counts.begin()));
  thrust::tabulate(
    rmm::exec_policy(stream_view),
    rank_count_pair_first,
    rank_count_pair_first + num_groups,
    detail::compute_group_id_count_pair_t<decltype(group_id_first)>{
      group_id_first, group_id_first + cuda::std::distance(tx_value_first, tx_value_last)});

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
    tx_key_first, cuda::proclaim_return_type<int>([key_to_group_id_op] __device__(auto key) {
      return key_to_group_id_op(key);
    }));
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream_view);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream_view);
  auto rank_count_pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_tx_dst_ranks.begin(), d_tx_value_counts.begin()));
  thrust::tabulate(
    rmm::exec_policy(stream_view),
    rank_count_pair_first,
    rank_count_pair_first + num_groups,
    detail::compute_group_id_count_pair_t<decltype(group_id_first)>{
      group_id_first, group_id_first + cuda::std::distance(tx_key_first, tx_key_last)});

  return d_tx_value_counts;
}

template <typename TxValueIterator>
auto shuffle_values(raft::comms::comms_t const& comm,
                    TxValueIterator tx_value_first,
                    raft::host_span<size_t const> tx_value_counts,
                    rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_tx_value_counts(comm_size, stream_view);
  raft::update_device(
    d_tx_value_counts.data(), tx_value_counts.data(), comm_size, stream_view.value());

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_displs{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_displs{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks) =
    detail::compute_tx_rx_counts_displs_ranks(comm, d_tx_value_counts, true, stream_view);

  auto rx_value_buffer = allocate_dataframe_buffer<value_t>(
    rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0}, stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
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

// Add gaps in the receive buffer to enforce that the sent data offset and the received data offset
// have the same alignment for every rank. This is faster assuming that @p alignment ensures cache
// line alignment in both send & receive buffer (tested with NCCL 2.23.4)
template <typename TxValueIterator>
auto shuffle_values(
  raft::comms::comms_t const& comm,
  TxValueIterator tx_value_first,
  raft::host_span<size_t const> tx_value_counts,
  size_t alignment,  // # elements
  std::optional<typename thrust::iterator_traits<TxValueIterator>::value_type> fill_value,
  rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  auto const comm_size = comm.get_size();

  std::vector<size_t> tx_value_displacements(tx_value_counts.size());
  std::exclusive_scan(
    tx_value_counts.begin(), tx_value_counts.end(), tx_value_displacements.begin(), size_t{0});

  std::vector<size_t> tx_unaligned_counts(comm_size);
  std::vector<size_t> tx_displacements(comm_size);
  std::vector<size_t> tx_aligned_counts(comm_size);
  std::vector<size_t> tx_aligned_displacements(comm_size);
  std::vector<size_t> rx_unaligned_counts(comm_size);
  std::vector<size_t> rx_displacements(comm_size);
  std::vector<size_t> rx_aligned_counts(comm_size);
  std::vector<size_t> rx_aligned_displacements(comm_size);
  std::vector<int> tx_ranks(comm_size);
  std::iota(tx_ranks.begin(), tx_ranks.end(), int{0});
  auto rx_ranks = tx_ranks;
  for (size_t i = 0; i < tx_value_counts.size(); ++i) {
    tx_unaligned_counts[i] = 0;
    if (tx_value_displacements[i] % alignment != 0) {
      tx_unaligned_counts[i] =
        std::min(alignment - (tx_value_displacements[i] % alignment), tx_value_counts[i]);
    }
    tx_displacements[i]         = tx_value_displacements[i];
    tx_aligned_counts[i]        = tx_value_counts[i] - tx_unaligned_counts[i];
    tx_aligned_displacements[i] = tx_value_displacements[i] + tx_unaligned_counts[i];
  }

  rmm::device_uvector<size_t> d_tx_unaligned_counts(tx_unaligned_counts.size(), stream_view);
  rmm::device_uvector<size_t> d_tx_aligned_counts(tx_aligned_counts.size(), stream_view);
  rmm::device_uvector<size_t> d_rx_unaligned_counts(rx_unaligned_counts.size(), stream_view);
  rmm::device_uvector<size_t> d_rx_aligned_counts(rx_aligned_counts.size(), stream_view);
  raft::update_device(d_tx_unaligned_counts.data(),
                      tx_unaligned_counts.data(),
                      tx_unaligned_counts.size(),
                      stream_view);
  raft::update_device(
    d_tx_aligned_counts.data(), tx_aligned_counts.data(), tx_aligned_counts.size(), stream_view);
  std::vector<size_t> tx_counts(comm_size, size_t{1});
  std::vector<size_t> tx_displs(comm_size);
  std::iota(tx_displs.begin(), tx_displs.end(), size_t{0});
  auto rx_counts = tx_counts;
  auto rx_displs = tx_displs;
  cugraph::device_multicast_sendrecv(
    comm,
    d_tx_unaligned_counts.data(),
    raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
    raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    d_rx_unaligned_counts.data(),
    raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
    raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);
  cugraph::device_multicast_sendrecv(
    comm,
    d_tx_aligned_counts.data(),
    raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
    raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    d_rx_aligned_counts.data(),
    raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
    raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);
  raft::update_host(rx_unaligned_counts.data(),
                    d_rx_unaligned_counts.data(),
                    d_rx_unaligned_counts.size(),
                    stream_view);
  raft::update_host(
    rx_aligned_counts.data(), d_rx_aligned_counts.data(), d_rx_aligned_counts.size(), stream_view);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view));
  size_t offset{0};
  for (size_t i = 0; i < rx_counts.size(); ++i) {
    auto target_alignment = (alignment - rx_unaligned_counts[i]) % alignment;
    auto cur_alignment    = offset % alignment;
    if (target_alignment >= cur_alignment) {
      offset += target_alignment - cur_alignment;
    } else {
      offset += (target_alignment + alignment) - cur_alignment;
    }
    rx_displacements[i]         = offset;
    rx_aligned_displacements[i] = rx_displacements[i] + rx_unaligned_counts[i];
    offset                      = rx_aligned_displacements[i] + rx_aligned_counts[i];
  }

  auto rx_values = allocate_dataframe_buffer<value_t>(
    rx_aligned_displacements.back() + rx_aligned_counts.back(), stream_view);
  if (fill_value) {
    thrust::fill(rmm::exec_policy_nosync(stream_view),
                 get_dataframe_buffer_begin(rx_values),
                 get_dataframe_buffer_end(rx_values),
                 *fill_value);
  }
  cugraph::device_multicast_sendrecv(
    comm,
    tx_value_first,
    raft::host_span<size_t const>(tx_unaligned_counts.data(), tx_unaligned_counts.size()),
    raft::host_span<size_t const>(tx_displacements.data(), tx_displacements.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    get_dataframe_buffer_begin(rx_values),
    raft::host_span<size_t const>(rx_unaligned_counts.data(), rx_unaligned_counts.size()),
    raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);
  cugraph::device_multicast_sendrecv(
    comm,
    tx_value_first,
    raft::host_span<size_t const>(tx_aligned_counts.data(), tx_aligned_counts.size()),
    raft::host_span<size_t const>(tx_aligned_displacements.data(), tx_aligned_displacements.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    get_dataframe_buffer_begin(rx_values),
    raft::host_span<size_t const>(rx_aligned_counts.data(), rx_aligned_counts.size()),
    raft::host_span<size_t const>(rx_aligned_displacements.data(), rx_aligned_displacements.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);

  return std::make_tuple(std::move(rx_values),
                         tx_unaligned_counts,
                         tx_aligned_counts,
                         tx_displacements,
                         rx_unaligned_counts,
                         rx_aligned_counts,
                         rx_displacements);
}

// this uses less memory than calling shuffle_values then sort & unique but requires comm.get_size()
// - 1 communication steps
template <typename TxValueIterator>
auto shuffle_and_unique_segment_sorted_values(
  raft::comms::comms_t const& comm,
  TxValueIterator
    segment_sorted_tx_value_first,  // sorted within each segment (segment sizes:
                                    // tx_value_counts[i], where i = [0, comm_size); and bettter be
                                    // unique to reduce communication volume
  raft::host_span<size_t const> tx_value_counts,
  rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  auto sorted_unique_values = allocate_dataframe_buffer<value_t>(0, stream_view);
  if (comm_size == 1) {
    resize_dataframe_buffer(sorted_unique_values, tx_value_counts[comm_rank], stream_view);
    thrust::copy(rmm::exec_policy_nosync(stream_view),
                 segment_sorted_tx_value_first,
                 segment_sorted_tx_value_first + tx_value_counts[comm_rank],
                 get_dataframe_buffer_begin(sorted_unique_values));
    resize_dataframe_buffer(
      sorted_unique_values,
      cuda::std::distance(get_dataframe_buffer_begin(sorted_unique_values),
                          thrust::unique(rmm::exec_policy_nosync(stream_view),
                                         get_dataframe_buffer_begin(sorted_unique_values),
                                         get_dataframe_buffer_end(sorted_unique_values))),
      stream_view);
  } else {
    rmm::device_uvector<size_t> d_tx_value_counts(comm_size, stream_view);
    raft::update_device(
      d_tx_value_counts.data(), tx_value_counts.data(), comm_size, stream_view.value());

    std::vector<size_t> tx_counts{};
    std::vector<size_t> tx_displs{};
    std::vector<size_t> rx_counts{};
    std::vector<size_t> rx_displs{};
    std::tie(tx_counts, tx_displs, std::ignore, rx_counts, rx_displs, std::ignore) =
      detail::compute_tx_rx_counts_displs_ranks(comm, d_tx_value_counts, false, stream_view);

    d_tx_value_counts.resize(0, stream_view);
    d_tx_value_counts.shrink_to_fit(stream_view);

    for (int i = 1; i < comm_size; ++i) {
      auto dst = (comm_rank + i) % comm_size;
      auto src =
        static_cast<int>((static_cast<size_t>(comm_rank) + static_cast<size_t>(comm_size - i)) %
                         static_cast<size_t>(comm_size));
      auto rx_sorted_values = allocate_dataframe_buffer<value_t>(rx_counts[src], stream_view);
      device_sendrecv(comm,
                      segment_sorted_tx_value_first + tx_displs[dst],
                      tx_counts[dst],
                      dst,
                      get_dataframe_buffer_begin(rx_sorted_values),
                      rx_counts[src],
                      src,
                      stream_view);
      auto merged_sorted_values = allocate_dataframe_buffer<value_t>(
        (i == 1 ? tx_counts[comm_rank] : size_dataframe_buffer(sorted_unique_values)) +
          rx_counts[src],
        stream_view);
      if (i == 1) {
        thrust::merge(rmm::exec_policy_nosync(stream_view),
                      segment_sorted_tx_value_first + tx_displs[comm_rank],
                      segment_sorted_tx_value_first + (tx_displs[comm_rank] + tx_counts[comm_rank]),
                      get_dataframe_buffer_begin(rx_sorted_values),
                      get_dataframe_buffer_end(rx_sorted_values),
                      get_dataframe_buffer_begin(merged_sorted_values));
      } else {
        thrust::merge(rmm::exec_policy_nosync(stream_view),
                      get_dataframe_buffer_begin(sorted_unique_values),
                      get_dataframe_buffer_end(sorted_unique_values),
                      get_dataframe_buffer_begin(rx_sorted_values),
                      get_dataframe_buffer_end(rx_sorted_values),
                      get_dataframe_buffer_begin(merged_sorted_values));
      }
      resize_dataframe_buffer(
        merged_sorted_values,
        cuda::std::distance(get_dataframe_buffer_begin(merged_sorted_values),
                            thrust::unique(rmm::exec_policy_nosync(stream_view),
                                           get_dataframe_buffer_begin(merged_sorted_values),
                                           get_dataframe_buffer_end(merged_sorted_values))),
        stream_view);
      sorted_unique_values = std::move(merged_sorted_values);
    }
  }
  shrink_to_fit_dataframe_buffer(sorted_unique_values, stream_view);
  return sorted_unique_values;
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
  std::vector<size_t> tx_displs{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_displs{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks) =
    detail::compute_tx_rx_counts_displs_ranks(comm, d_tx_value_counts, true, stream_view);

  auto rx_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0}, stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
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
  std::vector<size_t> tx_displs{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_displs{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks) =
    detail::compute_tx_rx_counts_displs_ranks(comm, d_tx_value_counts, true, stream_view);

  rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type> rx_keys(
    rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0}, stream_view);
  auto rx_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      rx_keys.size(), stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_key_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            rx_keys.begin(),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
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

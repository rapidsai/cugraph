/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/mem_frugal_partition.cuh>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename GroupIdIterator>
struct compute_group_id_count_pair_t {
  GroupIdIterator group_id_first{};
  GroupIdIterator group_id_last{};

  __device__ cuda::std::tuple<int, size_t> operator()(size_t i) const
  {
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<GroupIdIterator>::value_type, int>);
    auto lower_it =
      thrust::lower_bound(thrust::seq, group_id_first, group_id_last, static_cast<int>(i));
    auto upper_it = thrust::upper_bound(thrust::seq, lower_it, group_id_last, static_cast<int>(i));
    return cuda::std::make_tuple(static_cast<int>(i),
                                 static_cast<size_t>(cuda::std::distance(lower_it, upper_it)));
  }
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
  __device__ bool operator()(cuda::std::tuple<key_type, value_type> t) const
  {
    return key_to_group_id_op(cuda::std::get<0>(t)) < pivot;
  }
};

template <typename gid_offset_t,
          typename offset_t,
          typename ValueIterator,
          typename ValueToGroupIdOp>
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
  rmm::device_uvector<gid_offset_t> group_id_offsets(num_values, stream_view);
  rmm::device_uvector<offset_t> intra_partition_displs(num_values, stream_view);
  cugraph::fill(rmm::exec_policy(stream_view), counts.begin(), counts.end(), size_t{0});
  thrust::transform(
    rmm::exec_policy(stream_view),
    value_first,
    value_last,
    thrust::make_zip_iterator(group_id_offsets.begin(), intra_partition_displs.begin()),
    cuda::proclaim_return_type<cuda::std::tuple<gid_offset_t, offset_t>>(
      [value_to_group_id_op, group_first, counts = counts.data()] __device__(auto value) {
        auto group_id_offset = static_cast<gid_offset_t>(value_to_group_id_op(value) - group_first);
        cuda::std::atomic_ref<size_t> counter(counts[group_id_offset]);
        return cuda::std::make_tuple(
          group_id_offset,
          static_cast<offset_t>(counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed)));
      }));

  rmm::device_uvector<size_t> displacements(num_groups, stream_view);
  cugraph::exclusive_scan(
    rmm::exec_policy(stream_view), counts.begin(), counts.end(), displacements.begin());

  auto tmp_value_buffer =
    allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
      num_values, stream_view);
  auto tmp_value_first = get_dataframe_buffer_begin(tmp_value_buffer);
  thrust::scatter(
    rmm::exec_policy(stream_view),
    value_first,
    value_last,
    cuda::make_transform_iterator(
      thrust::make_zip_iterator(group_id_offsets.begin(), intra_partition_displs.begin()),
      cuda::proclaim_return_type<size_t>(
        [displacements = raft::device_span<size_t const>(
           displacements.data(), displacements.size())] __device__(auto pair) {
          return displacements[cuda::std::get<0>(pair)] +
                 static_cast<size_t>(cuda::std::get<1>(pair));
        })),
    tmp_value_first);
  thrust::copy(
    rmm::exec_policy(stream_view), tmp_value_first, tmp_value_first + num_values, value_first);
}

template <typename gid_offset_t,
          typename offset_t,
          typename KeyIterator,
          typename ValueIterator,
          typename KeyToGroupIdOp>
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
  rmm::device_uvector<gid_offset_t> group_id_offsets(num_keys, stream_view);
  rmm::device_uvector<offset_t> intra_partition_displs(num_keys, stream_view);
  cugraph::fill(rmm::exec_policy(stream_view), counts.begin(), counts.end(), size_t{0});
  thrust::transform(
    rmm::exec_policy(stream_view),
    key_first,
    key_last,
    thrust::make_zip_iterator(group_id_offsets.begin(), intra_partition_displs.begin()),
    cuda::proclaim_return_type<cuda::std::tuple<gid_offset_t, offset_t>>(
      [key_to_group_id_op, group_first, counts = counts.data()] __device__(auto key) {
        auto group_id_offset = static_cast<gid_offset_t>(key_to_group_id_op(key) - group_first);
        cuda::std::atomic_ref<size_t> counter(counts[group_id_offset]);
        return cuda::std::make_tuple(
          group_id_offset,
          static_cast<offset_t>(counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed)));
      }));

  rmm::device_uvector<size_t> displacements(num_groups, stream_view);
  cugraph::exclusive_scan(
    rmm::exec_policy(stream_view), counts.begin(), counts.end(), displacements.begin());

  auto map_first = cuda::make_transform_iterator(
    thrust::make_zip_iterator(group_id_offsets.begin(), intra_partition_displs.begin()),
    cuda::proclaim_return_type<size_t>([displacements = raft::device_span<size_t const>(
                                          displacements.data(),
                                          displacements.size())] __device__(auto pair) {
      return displacements[cuda::std::get<0>(pair)] + static_cast<size_t>(cuda::std::get<1>(pair));
    }));
  {
    auto tmp_key_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<KeyIterator>::value_type>(
        num_keys, stream_view);
    auto tmp_key_first = get_dataframe_buffer_begin(tmp_key_buffer);
    thrust::scatter(rmm::exec_policy(stream_view), key_first, key_last, map_first, tmp_key_first);
    thrust::copy(rmm::exec_policy(stream_view), tmp_key_first, tmp_key_first + num_keys, key_first);
  }
  {
    auto tmp_value_buffer =
      allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
        num_keys, stream_view);
    auto tmp_value_first = get_dataframe_buffer_begin(tmp_value_buffer);
    thrust::scatter(rmm::exec_policy(stream_view),
                    value_first,
                    value_first + num_keys,
                    map_first,
                    tmp_value_first);
    thrust::copy(
      rmm::exec_policy(stream_view), tmp_value_first, tmp_value_first + num_keys, value_first);
  }
}

template <typename ValueIterator>
void swap_partitions(ValueIterator value_first,
                     ValueIterator value_last,
                     size_t first_partition_size,
                     rmm::cuda_stream_view stream_view,
                     std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto num_elements          = static_cast<size_t>(cuda::std::distance(value_first, value_last));
  auto second_partition_size = num_elements - first_partition_size;
  if (first_partition_size >= second_partition_size) {
    auto tmp_value_buffer =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<value_t>(first_partition_size, stream_view)
        : allocate_dataframe_buffer<value_t>(first_partition_size, stream_view);

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
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<value_t>(second_partition_size, stream_view)
        : allocate_dataframe_buffer<value_t>(second_partition_size, stream_view);

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
                     rmm::cuda_stream_view stream_view,
                     std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using key_t   = typename thrust::iterator_traits<KeyIterator>::value_type;
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto num_elements          = static_cast<size_t>(cuda::std::distance(key_first, key_last));
  auto second_partition_size = num_elements - first_partition_size;
  if (first_partition_size >= second_partition_size) {
    auto tmp_key_buffer =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<key_t>(first_partition_size, stream_view)
        : allocate_dataframe_buffer<key_t>(first_partition_size, stream_view);
    auto tmp_value_buffer =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<value_t>(first_partition_size, stream_view)
        : allocate_dataframe_buffer<value_t>(first_partition_size, stream_view);

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
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<key_t>(second_partition_size, stream_view)
        : allocate_dataframe_buffer<key_t>(second_partition_size, stream_view);
    auto tmp_value_buffer =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<value_t>(second_partition_size, stream_view)
        : allocate_dataframe_buffer<value_t>(second_partition_size, stream_view);

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

template <typename ValueIterator, typename ValueToGroupIdOp>
void mem_frugal_groupby(
  ValueIterator value_first,
  ValueIterator value_last,
  ValueToGroupIdOp value_to_group_id_op,
  int num_groups,
  size_t mem_frugal_threshold,  // take the memory frugal approach (instead of thrust::sort) if #
                                // elements to groupby is no smaller than this value
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

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
          if ((((group_lasts[i] - group_firsts[i]) - int{1}) <=
               static_cast<int>(std::numeric_limits<uint8_t>::max())) &&
              ((static_cast<size_t>(cuda::std::distance(value_firsts[i], value_lasts[i])) -
                size_t{1}) <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()))) {
            multi_partition<uint8_t, uint32_t>(value_firsts[i],
                                               value_lasts[i],
                                               value_to_group_id_op,
                                               group_firsts[i],
                                               group_lasts[i],
                                               stream_view);
          } else {
            multi_partition<int, size_t>(value_firsts[i],
                                         value_lasts[i],
                                         value_to_group_id_op,
                                         group_firsts[i],
                                         group_lasts[i],
                                         stream_view);
          }
        }
      } else {
        ValueIterator second_first{};
        auto num_elements =
          static_cast<size_t>(cuda::std::distance(value_firsts[i], value_lasts[i]));
        auto first_chunk_partition_first  = mem_frugal_partition(value_firsts[i],
                                                                value_firsts[i] + num_elements / 2,
                                                                value_to_group_id_op,
                                                                pivot,
                                                                stream_view,
                                                                large_buffer_type);
        auto second_chunk_partition_first = mem_frugal_partition(value_firsts[i] + num_elements / 2,
                                                                 value_lasts[i],
                                                                 value_to_group_id_op,
                                                                 pivot,
                                                                 stream_view,
                                                                 large_buffer_type);
        auto no_less_size                 = static_cast<size_t>(
          cuda::std::distance(first_chunk_partition_first, value_firsts[i] + num_elements / 2));
        auto less_size = static_cast<size_t>(
          cuda::std::distance(value_firsts[i] + num_elements / 2, second_chunk_partition_first));
        swap_partitions(value_firsts[i] + (num_elements / 2 - no_less_size),
                        value_firsts[i] + (num_elements / 2 + less_size),
                        no_less_size,
                        stream_view,
                        large_buffer_type);

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
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

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
          auto kv_pair_first = thrust::make_zip_iterator(key_firsts[i], value_firsts[i]);
          thrust::partition(
            rmm::exec_policy(stream_view),
            kv_pair_first,
            kv_pair_first + cuda::std::distance(key_firsts[i], key_lasts[i]),
            kv_pair_group_id_less_t<typename thrust::iterator_traits<KeyIterator>::value_type,
                                    typename thrust::iterator_traits<ValueIterator>::value_type,
                                    KeyToGroupIdOp>{key_to_group_id_op, pivot});
        } else {
          if ((((group_lasts[i] - group_firsts[i]) - int{1}) <=
               static_cast<int>(std::numeric_limits<uint8_t>::max())) &&
              ((static_cast<size_t>(cuda::std::distance(key_firsts[i], key_lasts[i])) -
                size_t{1}) <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()))) {
            multi_partition<uint8_t, uint32_t>(key_firsts[i],
                                               key_lasts[i],
                                               value_firsts[i],
                                               key_to_group_id_op,
                                               group_firsts[i],
                                               group_lasts[i],
                                               stream_view);
          } else {
            multi_partition<int, size_t>(key_firsts[i],
                                         key_lasts[i],
                                         value_firsts[i],
                                         key_to_group_id_op,
                                         group_firsts[i],
                                         group_lasts[i],
                                         stream_view);
          }
        }
      } else {
        std::tuple<KeyIterator, ValueIterator> second_first{};
        auto num_elements = static_cast<size_t>(cuda::std::distance(key_firsts[i], key_lasts[i]));
        auto first_chunk_partition_first  = mem_frugal_partition(key_firsts[i],
                                                                key_firsts[i] + num_elements / 2,
                                                                value_firsts[i],
                                                                key_to_group_id_op,
                                                                pivot,
                                                                stream_view,
                                                                large_buffer_type);
        auto second_chunk_partition_first = mem_frugal_partition(key_firsts[i] + num_elements / 2,
                                                                 key_lasts[i],
                                                                 value_firsts[i] + num_elements / 2,
                                                                 key_to_group_id_op,
                                                                 pivot,
                                                                 stream_view,
                                                                 large_buffer_type);
        auto no_less_size                 = static_cast<size_t>(cuda::std::distance(
          std::get<0>(first_chunk_partition_first), key_firsts[i] + num_elements / 2));
        auto less_size                    = static_cast<size_t>(cuda::std::distance(
          key_firsts[i] + num_elements / 2, std::get<0>(second_chunk_partition_first)));
        swap_partitions(key_firsts[i] + (num_elements / 2 - no_less_size),
                        key_firsts[i] + (num_elements / 2 + less_size),
                        value_firsts[i] + (num_elements / 2 - no_less_size),
                        no_less_size,
                        stream_view,
                        large_buffer_type);

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
rmm::device_uvector<size_t> groupby_and_count(
  ValueIterator tx_value_first /* [INOUT */,
  ValueIterator tx_value_last /* [INOUT */,
  ValueToGroupIdOp value_to_group_id_op,
  int num_groups,
  size_t mem_frugal_threshold,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  detail::mem_frugal_groupby(tx_value_first,
                             tx_value_last,
                             value_to_group_id_op,
                             num_groups,
                             mem_frugal_threshold,
                             stream_view,
                             large_buffer_type);

  auto group_id_first = cuda::make_transform_iterator(
    tx_value_first, cuda::proclaim_return_type<int>([value_to_group_id_op] __device__(auto value) {
      return value_to_group_id_op(value);
    }));
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream_view);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream_view);
  auto rank_count_pair_first =
    thrust::make_zip_iterator(d_tx_dst_ranks.begin(), d_tx_value_counts.begin());
  thrust::tabulate(
    rmm::exec_policy(stream_view),
    rank_count_pair_first,
    rank_count_pair_first + num_groups,
    detail::compute_group_id_count_pair_t<decltype(group_id_first)>{
      group_id_first, group_id_first + cuda::std::distance(tx_value_first, tx_value_last)});

  return d_tx_value_counts;
}

template <typename KeyIterator, typename ValueIterator, typename KeyToGroupIdOp>
rmm::device_uvector<size_t> groupby_and_count(
  KeyIterator tx_key_first /* [INOUT */,
  KeyIterator tx_key_last /* [INOUT */,
  ValueIterator tx_value_first /* [INOUT */,
  KeyToGroupIdOp key_to_group_id_op,
  int num_groups,
  size_t mem_frugal_threshold,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  detail::mem_frugal_groupby(tx_key_first,
                             tx_key_last,
                             tx_value_first,
                             key_to_group_id_op,
                             num_groups,
                             mem_frugal_threshold,
                             stream_view,
                             large_buffer_type);

  auto group_id_first = cuda::make_transform_iterator(
    tx_key_first, cuda::proclaim_return_type<int>([key_to_group_id_op] __device__(auto key) {
      return key_to_group_id_op(key);
    }));
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream_view);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream_view);
  auto rank_count_pair_first =
    thrust::make_zip_iterator(d_tx_dst_ranks.begin(), d_tx_value_counts.begin());
  thrust::tabulate(
    rmm::exec_policy(stream_view),
    rank_count_pair_first,
    rank_count_pair_first + num_groups,
    detail::compute_group_id_count_pair_t<decltype(group_id_first)>{
      group_id_first, group_id_first + cuda::std::distance(tx_key_first, tx_key_last)});

  return d_tx_value_counts;
}

}  // namespace CUGRAPH_EXPORT cugraph

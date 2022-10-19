/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/atomic_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>

#include <raft/core/device_span.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename T>
class edge_partition_endpoint_property_device_view_t {
 public:
  using value_type = std::remove_const_t<T>;
  using buffer_type =
    std::conditional_t<std::is_same_v<value_type, bool>,
                       decltype(allocate_dataframe_buffer<uint32_t>(0, rmm::cuda_stream_view{})),
                       decltype(allocate_dataframe_buffer<value_type>(0, rmm::cuda_stream_view{}))>;
  using value_iterator = std::conditional_t<
    std::is_same_v<value_type, bool>,
    std::conditional_t<std::is_const_v<T>, uint32_t const*, uint32_t*>,
    std::conditional_t<
      std::is_const_v<T>,
      std::invoke_result_t<decltype(get_dataframe_buffer_cbegin<buffer_type>), buffer_type&>,
      std::invoke_result_t<decltype(get_dataframe_buffer_begin<buffer_type>), buffer_type&>>>;

  edge_partition_endpoint_property_device_view_t() = default;

  edge_partition_endpoint_property_device_view_t(
    edge_major_property_view_t<vertex_t, T> const& view, size_t partition_idx)
    : value_first_(view.value_firsts()[partition_idx]),
      range_first_(view.major_range_firsts()[partition_idx])
  {
    if (view.keys()) {
      keys_                    = (*(view.keys()))[partition_idx];
      key_chunk_start_offsets_ = (*(view.key_chunk_start_offsets()))[partition_idx];
      key_chunk_size_          = *(view.key_chunk_size());
    }
    value_first_ = view.value_firsts()[partition_idx];
    range_first_ = view.major_range_firsts()[partition_idx];
  }

  edge_partition_endpoint_property_device_view_t(
    edge_minor_property_view_t<vertex_t, T> const& view)
  {
    if (view.keys()) {
      keys_                    = *(view.keys());
      key_chunk_start_offsets_ = *(view.key_chunk_start_offsets());
      key_chunk_size_          = *(view.key_chunk_size());
    }
    value_first_ = view.value_first();
    range_first_ = view.minor_range_first();
  }

  __device__ value_type get(vertex_t offset) const
  {
    auto val_offset = value_offset(offset);
    if constexpr (std::is_same_v<value_type, bool>) {
      auto mask = uint32_t{1} << (val_offset % (sizeof(uint32_t) * 8));
      return static_cast<bool>(*(value_first_ + (val_offset / (sizeof(uint32_t) * 8))) & mask);
    } else {
      return *(value_first_ + val_offset);
    }
  }

  template <typename Tp = T>
  __device__ std::enable_if_t<!std::is_const_v<Tp>, value_type> atomic_cas(vertex_t offset,
                                                                           value_type compare,
                                                                           value_type val) const
  {
    auto val_offset = value_offset(offset);
    if constexpr (std::is_same_v<value_type, bool>) {
      auto mask = uint32_t{1} << (val_offset % (sizeof(uint32_t) * 8));
      auto old  = val ? atomicOr(value_first_ + (val_offset / (sizeof(uint32_t) * 8)), mask)
                      : atomicAnd(value_first_ + (val_offset / (sizeof(uint32_t) * 8)), ~mask);
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::atomic_cas(value_first_ + val_offset, compare, val);
    }
  }

  template <typename Tp = T>
  __device__ std::enable_if_t<!std::is_const_v<Tp>, value_type> atomic_or(vertex_t offset,
                                                                          value_type val) const
  {
    auto val_offset = value_offset(offset);
    if constexpr (std::is_same_v<value_type, bool>) {
      auto mask = uint32_t{1} << (val_offset % (sizeof(uint32_t) * 8));
      auto old =
        atomicOr(value_first_ + (val_offset / (sizeof(uint32_t) * 8)), val ? mask : uint32_t{0});
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::atomic_or(value_first_ + val_offset, val);
    }
  }

  template <typename Tp = T>
  __device__ std::enable_if_t<!std::is_const_v<Tp> &&
                                !std::is_same_v<Tp, bool> /* accumulation undefined for bool */,
                              void>
  atomic_accumulate(vertex_t offset, value_type val) const
  {
    auto val_offset = value_offset(offset);
    cugraph::atomic_accumulate(value_first_ + val_offset, val);
  }

 private:
  thrust::optional<raft::device_span<vertex_t const>> keys_{thrust::nullopt};
  thrust::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{thrust::nullopt};
  thrust::optional<size_t> key_chunk_size_{thrust::nullopt};

  value_iterator value_first_{};
  vertex_t range_first_{};

  __device__ vertex_t value_offset(vertex_t offset) const
  {
    auto val_offset = offset;
    if (keys_) {
      auto chunk_idx = static_cast<size_t>(offset) / (*key_chunk_size_);
      auto it        = thrust::lower_bound(thrust::seq,
                                    (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx],
                                    (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1],
                                    range_first_ + offset);
      assert((it != (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1]) &&
             (*it == (range_first_ + offset)));
      val_offset = (*key_chunk_start_offsets_)[chunk_idx] +
                   static_cast<vertex_t>(thrust::distance(
                     (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx], it));
    }
    return val_offset;
  }
};

template <typename vertex_t>
class edge_partition_endpoint_dummy_property_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  edge_partition_endpoint_dummy_property_device_view_t() = default;

  edge_partition_endpoint_dummy_property_device_view_t(
    edge_endpoint_dummy_property_view_t const& view, size_t partition_idx)
  {
  }

  edge_partition_endpoint_dummy_property_device_view_t(
    edge_endpoint_dummy_property_view_t const& view)
  {
  }

  __device__ auto get(vertex_t offset) const { return thrust::nullopt; }
};

}  // namespace detail

}  // namespace cugraph

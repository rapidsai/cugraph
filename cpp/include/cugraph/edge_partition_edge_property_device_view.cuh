/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_property.hpp>
#include <cugraph/utilities/atomic_ops.cuh>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace cugraph {

namespace detail {

template <typename edge_t, typename ValueIterator, bool packed_bool = false>
class edge_partition_edge_property_device_view_t {
 public:
  static_assert(
    !packed_bool ||
    std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, uint32_t>);

  using edge_type  = edge_t;
  using value_type = std::
    conditional_t<packed_bool, bool, typename thrust::iterator_traits<ValueIterator>::value_type>;

  edge_partition_edge_property_device_view_t() = default;

  edge_partition_edge_property_device_view_t(
    edge_property_view_t<edge_t, ValueIterator> const& view, size_t partition_idx)
    : value_first_(view.value_firsts()[partition_idx])
  {
    value_first_ = view.value_firsts()[partition_idx];
  }

  __host__ __device__ ValueIterator value_first() { return value_first_; }

  __device__ value_type get(edge_t offset) const
  {
    if constexpr (packed_bool) {
      auto mask = uint32_t{1} << (offset % (sizeof(uint32_t) * 8));
      return static_cast<bool>(*(value_first_ + (offset / (sizeof(uint32_t) * 8))) & mask);
    } else {
      return *(value_first_ + offset);
    }
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_type>
  atomic_and(edge_t offset, value_type val) const
  {
    if constexpr (std::is_same_v<value_type, bool>) {
      auto mask = uint32_t{1} << (offset % (sizeof(uint32_t) * 8));
      auto old  = atomicAnd(value_first_ + (offset / (sizeof(uint32_t) * 8)),
                           val ? uint32_t{0xffffffff} : ~mask);
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::atomic_and(value_first_ + offset, val);
    }
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_type>
  atomic_or(edge_t offset, value_type val) const
  {
    if constexpr (std::is_same_v<value_type, bool>) {
      auto mask = uint32_t{1} << (offset % (sizeof(uint32_t) * 8));
      auto old =
        atomicOr(value_first_ + (offset / (sizeof(uint32_t) * 8)), val ? mask : uint32_t{0});
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::atomic_or(value_first_ + offset, val);
    }
  }

  template <typename Iter = ValueIterator, bool packed = packed_bool>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !packed /* add undefined for (packed-)bool */,
    value_type>
  atomic_add(edge_t offset, value_type val) const
  {
    cugraph::atomic_add(value_first_ + offset, val);
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_type>
  elementwise_atomic_cas(edge_t offset, value_type compare, value_type val) const
  {
    if constexpr (packed_bool) {
      auto mask = uint32_t{1} << (offset % (sizeof(uint32_t) * 8));
      auto old  = val ? atomicOr(value_first_ + (offset / (sizeof(uint32_t) * 8)), mask)
                      : atomicAnd(value_first_ + (offset / (sizeof(uint32_t) * 8)), ~mask);
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::elementwise_atomic_cas(value_first_ + offset, compare, val);
    }
  }

  template <typename Iter = ValueIterator, bool packed = packed_bool>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !packed /* min undefined for (packed-)bool */,
    value_type>
  elementwise_atomic_min(edge_t offset, value_type val) const
  {
    cugraph::elementwise_atomic_min(value_first_ + offset, val);
  }

  template <typename Iter = ValueIterator, bool packed = packed_bool>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !packed /* max undefined for (packed-)bool */,
    value_type>
  elementwise_atomic_max(edge_t offset, value_type val) const
  {
    cugraph::elementwise_atomic_max(value_first_ + offset, val);
  }

 private:
  ValueIterator value_first_{};
};

template <typename edge_t>
class edge_partition_edge_dummy_property_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  edge_partition_edge_dummy_property_device_view_t() = default;

  edge_partition_edge_dummy_property_device_view_t(edge_dummy_property_view_t const& view,
                                                   size_t partition_idx)
  {
  }

  __device__ auto get(edge_t offset) const { return thrust::nullopt; }
};

}  // namespace detail

}  // namespace cugraph

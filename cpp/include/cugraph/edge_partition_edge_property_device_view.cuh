/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace cugraph {

namespace detail {

template <typename edge_t,
          typename ValueIterator,
          typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
class edge_partition_edge_property_device_view_t {
 public:
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t> ||
    cugraph::has_packed_bool_element<ValueIterator, value_t>());
  static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

  using edge_type  = edge_t;
  using value_type = value_t;

  edge_partition_edge_property_device_view_t() = default;

  edge_partition_edge_property_device_view_t(
    edge_property_view_t<edge_t, ValueIterator> const& view, size_t partition_idx)
    : value_first_(view.value_firsts()[partition_idx])
  {
    value_first_ = view.value_firsts()[partition_idx];
  }

  __host__ __device__ ValueIterator value_first() { return value_first_; }

  __device__ value_t get(edge_t offset) const
  {
    if constexpr (cugraph::has_packed_bool_element<ValueIterator, value_t>()) {
      static_assert(std::is_arithmetic_v<value_t>, "unimplemented for thrust::tuple types.");
      auto mask = cugraph::packed_bool_mask(offset);
      return static_cast<bool>(*(value_first_ + cugraph::packed_bool_offset(offset)) & mask);
    } else {
      return *(value_first_ + offset);
    }
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    void>
  set(edge_t offset, value_t val) const
  {
    if constexpr (cugraph::has_packed_bool_element<ValueIterator, value_t>()) {
      static_assert(std::is_arithmetic_v<value_t>, "unimplemented for thrust::tuple types.");
      auto mask = cugraph::packed_bool_mask(offset);
      if (val) {
        atomicOr(value_first_ + cugraph::packed_bool_offset(offset), mask);
      } else {
        atomicAnd(value_first_ + cugraph::packed_bool_offset(offset), ~mask);
      }
    } else {
      *(value_first_ + offset) = val;
    }
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_t>
  atomic_and(edge_t offset, value_t val) const
  {
    if constexpr (cugraph::has_packed_bool_element<ValueIterator, value_t>()) {
      static_assert(std::is_arithmetic_v<value_t>, "unimplemented for thrust::tuple types.");
      auto mask = cugraph::packed_bool_mask(offset);
      auto old  = atomicAnd(value_first_ + cugraph::packed_bool_offset(offset),
                           val ? uint32_t{0xffffffff} : ~mask);
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::atomic_and(value_first_ + offset, val);
    }
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_t>
  atomic_or(edge_t offset, value_t val) const
  {
    if constexpr (cugraph::has_packed_bool_element<ValueIterator, value_t>()) {
      static_assert(std::is_arithmetic_v<value_t>, "unimplemented for thrust::tuple types.");
      auto mask = cugraph::packed_bool_mask(offset);
      auto old =
        atomicOr(value_first_ + cugraph::packed_bool_offset(offset), val ? mask : uint32_t{0});
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::atomic_or(value_first_ + offset, val);
    }
  }

  template <typename Iter = ValueIterator, typename T = value_t>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !cugraph::has_packed_bool_element<Iter, T>() /* add undefined for (packed-)bool */,
    value_t>
  atomic_add(edge_t offset, value_t val) const
  {
    cugraph::atomic_add(value_first_ + offset, val);
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_t>
  elementwise_atomic_cas(edge_t offset, value_t compare, value_t val) const
  {
    if constexpr (cugraph::has_packed_bool_element<ValueIterator, value_t>()) {
      static_assert(std::is_arithmetic_v<value_t>, "unimplemented for thrust::tuple types.");
      auto mask = cugraph::packed_bool_mask(offset);
      auto old  = val ? atomicOr(value_first_ + cugraph::packed_bool_offset(offset), mask)
                      : atomicAnd(value_first_ + cugraph::packed_bool_offset(offset), ~mask);
      return static_cast<bool>(old & mask);
    } else {
      return cugraph::elementwise_atomic_cas(value_first_ + offset, compare, val);
    }
  }

  template <typename Iter = ValueIterator, typename T = value_t>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !cugraph::has_packed_bool_element<Iter, T>() /* min undefined for (packed-)bool */,
    value_t>
  elementwise_atomic_min(edge_t offset, value_t val) const
  {
    cugraph::elementwise_atomic_min(value_first_ + offset, val);
  }

  template <typename Iter = ValueIterator, typename T = value_t>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !cugraph::has_packed_bool_element<Iter, T>() /* max undefined for (packed-)bool */,
    value_t>
  elementwise_atomic_max(edge_t offset, value_t val) const
  {
    cugraph::elementwise_atomic_max(value_first_ + offset, val);
  }

 private:
  ValueIterator value_first_{};
};

template <typename edge_t>
class edge_partition_edge_dummy_property_device_view_t {
 public:
  using edge_type  = edge_t;
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/edge_property.hpp>
#include <cugraph/utilities/atomic_ops.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>

#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/iterator/iterator_traits.h>

namespace cugraph {

namespace detail {

template <typename edge_t,
          typename ValueIterator,
          typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
class edge_partition_edge_property_device_view_t {
 public:
  using edge_type  = edge_t;
  using value_type = value_t;

  static constexpr bool is_packed_bool = cugraph::is_packed_bool<ValueIterator, value_t>();
  static constexpr bool has_packed_bool_element =
    cugraph::has_packed_bool_element<ValueIterator, value_t>();

  static_assert(
    std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t> ||
    has_packed_bool_element);
  static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

  edge_partition_edge_property_device_view_t() = default;

  edge_partition_edge_property_device_view_t(
    edge_property_view_t<edge_t, ValueIterator, value_t> const& view, size_t partition_idx)
    : value_first_(view.value_firsts()[partition_idx])
  {
  }

  __host__ __device__ ValueIterator value_first() const { return value_first_; }

  __device__ value_t get(edge_t offset) const
  {
    if constexpr (has_packed_bool_element) {
      static_assert(is_packed_bool, "unimplemented for cuda::std::tuple types.");
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
    if constexpr (has_packed_bool_element) {
      static_assert(is_packed_bool, "unimplemented for cuda::std::tuple types.");
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
    if constexpr (has_packed_bool_element) {
      static_assert(is_packed_bool, "unimplemented for cuda::std::tuple types.");
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
    if constexpr (has_packed_bool_element) {
      static_assert(is_packed_bool, "unimplemented for cuda::std::tuple types.");
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
    return cugraph::atomic_add(value_first_ + offset, val);
  }

  template <typename Iter = ValueIterator>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>>,
    value_t>
  elementwise_atomic_cas(edge_t offset, value_t compare, value_t val) const
  {
    if constexpr (has_packed_bool_element) {
      static_assert(is_packed_bool, "unimplemented for cuda::std::tuple types.");
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
        *(value_first_ + cugraph::packed_bool_offset(offset)));
      auto mask = cugraph::packed_bool_mask(offset);
      uint32_t old{};
      if (compare == val) {
        old = word.load(cuda::std::memory_order_relaxed);
      } else {
        old = val ? word.fetch_or(mask, cuda::std::memory_order_relaxed)
                  : word.fetch_and(~mask, cuda::std::memory_order_relaxed);
      }
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
    return cugraph::elementwise_atomic_min(value_first_ + offset, val);
  }

  template <typename Iter = ValueIterator, typename T = value_t>
  __device__ std::enable_if_t<
    !std::is_const_v<std::remove_reference_t<typename std::iterator_traits<Iter>::reference>> &&
      !cugraph::has_packed_bool_element<Iter, T>() /* max undefined for (packed-)bool */,
    value_t>
  elementwise_atomic_max(edge_t offset, value_t val) const
  {
    return cugraph::elementwise_atomic_max(value_first_ + offset, val);
  }

 private:
  ValueIterator value_first_{};
};

template <typename edge_t, typename vertex_t>
class edge_partition_edge_multi_index_property_device_view_t {
 public:
  using edge_type  = edge_t;
  using value_type = edge_t;

  static constexpr bool is_packed_bool          = false;
  static constexpr bool has_packed_bool_element = false;

  edge_partition_edge_multi_index_property_device_view_t() = default;

  edge_partition_edge_multi_index_property_device_view_t(
    edge_multi_index_property_view_t<edge_t, vertex_t> const& view, size_t partition_idx)
    : offsets_(view.offsets()[partition_idx]), indices_(view.indices()[partition_idx])
  {
  }

  __device__ edge_t get(edge_t offset) const
  {
    auto major_idx = static_cast<vertex_t>(cuda::std::distance(
      offsets_.begin() + 1,
      thrust::upper_bound(thrust::seq, offsets_.begin() + 1, offsets_.end(), offset)));
    auto nbr       = indices_[offset];
    return static_cast<edge_t>(
      cuda::std::distance(thrust::lower_bound(thrust::seq,
                                              indices_.begin() + offsets_[major_idx],
                                              indices_.begin() + offsets_[major_idx + 1],
                                              nbr),
                          indices_.begin() + offset));
  }

 private:
  raft::device_span<edge_t const> offsets_{};
  raft::device_span<vertex_t const> indices_{};
};

template <typename edge_t>
class edge_partition_edge_dummy_property_device_view_t {
 public:
  using edge_type  = edge_t;
  using value_type = cuda::std::nullopt_t;

  static constexpr bool is_packed_bool          = false;
  static constexpr bool has_packed_bool_element = false;

  edge_partition_edge_dummy_property_device_view_t() = default;

  edge_partition_edge_dummy_property_device_view_t(edge_dummy_property_view_t const& view,
                                                   size_t partition_idx)
  {
  }

  __device__ auto get(edge_t offset) const { return cuda::std::nullopt; }
};

}  // namespace detail

}  // namespace cugraph

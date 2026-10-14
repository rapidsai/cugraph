/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/optional>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace CUGRAPH_EXPORT cugraph {

template <typename vertex_t,
          typename ValueIterator,
          typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
class vertex_property_view_t {
 public:
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t> ||
    cugraph::has_packed_bool_element<ValueIterator, value_t>());

  using vertex_type    = vertex_t;
  using value_type     = value_t;
  using value_iterator = ValueIterator;

  vertex_property_view_t() = default;

  vertex_property_view_t(ValueIterator value_first, vertex_t range_size)
    : value_first_(value_first), range_size_(range_size)
  {
  }

  ValueIterator value_first() const { return value_first_; }

  vertex_t range_size() const { return range_size_; }

 private:
  ValueIterator value_first_{};
  vertex_t range_size_{0};
};

class vertex_dummy_property_view_t {
 public:
  using value_type     = cuda::std::nullopt_t;
  using value_iterator = void*;
};

template <typename vertex_t, typename T>
class vertex_property_t {
 public:
  static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_type = vertex_t;
  using value_type  = T;
  using buffer_type =
    dataframe_buffer_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;

  explicit vertex_property_t(raft::handle_t const& handle)
    : range_size_(vertex_t{0}),
      value_buffer_(
        allocate_dataframe_buffer<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>(
          size_t{0}, handle.get_stream()))
  {
  }

  template <typename GraphViewType>
  vertex_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
    : range_size_(graph_view.local_vertex_partition_range_size()),
      value_buffer_(
        allocate_dataframe_buffer<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>(
          std::is_same_v<T, bool>
            ? packed_bool_size(static_cast<size_t>(graph_view.local_vertex_partition_range_size()))
            : static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
          handle.get_stream()))
  {
    static_assert(std::is_same_v<vertex_t, typename GraphViewType::vertex_type>);
  }

  vertex_property_t(buffer_type&& value_buffer, vertex_t range_size)
    : value_buffer_(std::move(value_buffer)), range_size_(range_size)
  {
    auto const buffer_size = size_dataframe_buffer(value_buffer_);
    if constexpr (std::is_same_v<T, bool>) {
      CUGRAPH_EXPECTS(buffer_size == packed_bool_size(static_cast<size_t>(range_size)),
                      "Invalid input argument: range_size does not match the packed bool "
                      "buffer size.");
    } else {
      CUGRAPH_EXPECTS(buffer_size == static_cast<size_t>(range_size),
                      "Invalid input argument: range_size does not match the buffer size.");
    }
  }

  buffer_type release_buffer()
  {
    range_size_ = vertex_t{0};
    return std::move(value_buffer_);
  }

  void clear(raft::handle_t const& handle)
  {
    resize_dataframe_buffer(value_buffer_, size_t{0}, handle.get_stream());
    shrink_to_fit_dataframe_buffer(value_buffer_, handle.get_stream());
    range_size_ = vertex_t{0};
  }

  auto view() const
  {
    using const_value_iterator = decltype(get_dataframe_buffer_cbegin(value_buffer_));
    return vertex_property_view_t<vertex_type, const_value_iterator, T>(
      get_dataframe_buffer_cbegin(value_buffer_), range_size_);
  }

  auto mutable_view()
  {
    using value_iterator = decltype(get_dataframe_buffer_begin(value_buffer_));
    return vertex_property_view_t<vertex_type, value_iterator, T>(
      get_dataframe_buffer_begin(value_buffer_), range_size_);
  }

 private:
  buffer_type value_buffer_;
  vertex_t range_size_{0};
};

class vertex_dummy_property_t {
 public:
  using value_type = cuda::std::nullopt_t;

  auto view() const { return vertex_dummy_property_view_t{}; }
};

template <typename GraphViewType, typename VertexValueType, typename Enable = void>
struct vertex_property_view_type;

template <typename GraphViewType, typename VertexValueType>
struct vertex_property_view_type<
  GraphViewType,
  VertexValueType,
  std::enable_if_t<std::is_same_v<VertexValueType, cuda::std::nullopt_t>>> {
  using value = vertex_dummy_property_view_t;
};

template <typename GraphViewType, typename VertexValueType>
struct vertex_property_view_type<
  GraphViewType,
  VertexValueType,
  std::enable_if_t<!std::is_same_v<VertexValueType, cuda::std::nullopt_t>>> {
  using value = decltype(vertex_property_t<typename GraphViewType::vertex_type, VertexValueType>(
                           raft::handle_t{})
                           .view());
};

template <typename GraphViewType, typename VertexValueType>
using vertex_property_view_type_t =
  typename vertex_property_view_type<GraphViewType, VertexValueType>::value;

template <typename vertex_t, typename... Iters, typename... Types>
auto view_concat(vertex_property_view_t<vertex_t, Iters, Types> const&... views)
{
  using concat_value_iterator = decltype(thrust::make_zip_iterator(
    thrust_tuple_cat(to_thrust_iterator_tuple(views.value_first())...)));
  using concat_value_type     = decltype(thrust_tuple_cat(to_thrust_tuple(Types{})...));

  auto const first_view = get_first_of_pack(views...);
  return vertex_property_view_t<vertex_t, concat_value_iterator, concat_value_type>(
    thrust::make_zip_iterator(thrust_tuple_cat(to_thrust_iterator_tuple(views.value_first())...)),
    first_view.range_size());
}

}  // namespace CUGRAPH_EXPORT cugraph

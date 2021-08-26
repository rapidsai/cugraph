/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>

#include <optional>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename ValueIterator>
struct key_to_value_t {
  thrust::optional<vertex_t const*> const key_first{};
  thrust::optional<vertex_t const*> const key_last{};
  ValueIterator const value_first{};

  __device__ typename thrust::iterator_traits<ValueIterator>::value_type operator()(
    vertex_t offset) const
  {
    if (key_first) {
      auto it = thrust::lower_bound(thrust::seq, *key_first, *key_last, offset);
      assert((it != *key_last) && (*it == offset));
      return *(value_first + thrust::distance(*key_first, it));
    } else {
      return *(value_first + offset);
    }
  }
};

template <typename vertex_t, typename T>
class major_properties_t {
 public:
  major_properties_t() : buffer_(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{})) {}

  major_properties_t(raft::handle_t const& handle, vertex_t buffer_size)
    : buffer_(allocate_dataframe_buffer<T>(buffer_size, handle.get_stream()))
  {
  }

  void fill(T value, rmm::cuda_stream_view stream)
  {
    thrust::fill(
      rmm::exec_policy(stream), value_data(), value_data() + size_dataframe_buffer<T>(buffer_), value);
  }

  auto begin() const { return get_dataframe_buffer_begin<T>(buffer_); }

  auto value_data() { return get_dataframe_buffer_begin<T>(buffer_); }

 private:
  decltype(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{})) buffer_;
};

template <typename vertex_t, typename T>
class minor_properties_t {
 public:
  minor_properties_t()
    : key_first_(std::nullopt),
      key_last_(std::nullopt),
      buffer_(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{}))
  {
  }

  minor_properties_t(raft::handle_t const& handle, vertex_t buffer_size)
    : key_first_(std::nullopt),
      key_last_(std::nullopt),
      buffer_(allocate_dataframe_buffer<T>(buffer_size, handle.get_stream()))
  {
  }

  minor_properties_t(raft::handle_t const& handle,
                     vertex_t const* key_first,
                     vertex_t const* key_last)
    : key_first_(key_first),
      key_last_(key_last),
      buffer_(
        allocate_dataframe_buffer<T>(thrust::distance(key_first, key_last), handle.get_stream()))
  {
  }

  void fill(T value, rmm::cuda_stream_view stream)
  {
    thrust::fill(
      rmm::exec_policy(stream), value_data(), value_data() + size_dataframe_buffer<T>(buffer_), value);
  }

  auto begin() const
  {
    auto value_first = get_dataframe_buffer_begin<T>(buffer_);
    return thrust::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      key_to_value_t<vertex_t, decltype(value_first)>{
        key_first_ ? thrust::make_optional(*key_first_) : thrust::nullopt,
        key_last_ ? thrust::make_optional(*key_last_) : thrust::nullopt,
        value_first});
  }

  auto value_data() { return get_dataframe_buffer_begin<T>(buffer_); }

 private:
  std::optional<vertex_t const*> key_first_{std::nullopt};
  std::optional<vertex_t const*> key_last_{std::nullopt};

  decltype(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{})) buffer_;
};

}  // namespace detail

template <typename GraphViewType, typename T, typename Enable = void>
class row_properties_t;

template <typename GraphViewType, typename T>
class row_properties_t<GraphViewType,
                       T,
                       std::enable_if_t<GraphViewType::is_adj_matrix_transposed>> {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  row_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    auto key_first = graph_view.get_local_sorted_unique_edge_row_begin();
    auto key_last  = graph_view.get_local_sorted_unique_edge_row_end();
    if (key_first) {
      properties_ = detail::minor_properties_t<typename GraphViewType::vertex_type, T>(
        handle, *key_first, *key_last);
    } else {
      properties_ = detail::minor_properties_t<typename GraphViewType::vertex_type, T>(
        handle, graph_view.get_number_of_local_adj_matrix_partition_rows());
    }
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto begin() const { return properties_.begin(); }
  auto value_data() { return properties_.value_data(); }

 private:
  detail::minor_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

template <typename GraphViewType, typename T>
class row_properties_t<GraphViewType,
                       T,
                       std::enable_if_t<!GraphViewType::is_adj_matrix_transposed>> {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  row_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    properties_ = detail::major_properties_t<typename GraphViewType::vertex_type, T>(
      handle, graph_view.get_number_of_local_adj_matrix_partition_rows());
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto begin() const { return properties_.begin(); }
  auto value_data() { return properties_.value_data(); }

 private:
  detail::major_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

template <typename GraphViewType, typename T, typename Enable = void>
class col_properties_t;

template <typename GraphViewType, typename T>
class col_properties_t<GraphViewType,
                       T,
                       std::enable_if_t<GraphViewType::is_adj_matrix_transposed>> {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  col_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    properties_ = detail::major_properties_t<typename GraphViewType::vertex_type, T>(
      handle, graph_view.get_number_of_local_adj_matrix_partition_cols());
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto begin() const { return properties_.begin(); }
  auto value_data() { return properties_.value_data(); }

 private:
  detail::major_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

template <typename GraphViewType, typename T>
class col_properties_t<GraphViewType,
                       T,
                       std::enable_if_t<!GraphViewType::is_adj_matrix_transposed>> {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  col_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    auto key_first = graph_view.get_local_sorted_unique_edge_col_begin();
    auto key_last  = graph_view.get_local_sorted_unique_edge_col_end();
    if (key_first) {
      properties_ = detail::minor_properties_t<typename GraphViewType::vertex_type, T>(
        handle, *key_first, *key_last);
    } else {
      properties_ = detail::minor_properties_t<typename GraphViewType::vertex_type, T>(
        handle, graph_view.get_number_of_local_adj_matrix_partition_cols());
    }
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto begin() const { return properties_.begin(); }
  auto value_data() { return properties_.value_data(); }

 private:
  detail::minor_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

class dummy_properties_t {
 public:
  using value_type = thrust::nullopt_t;

  auto begin() const { return thrust::make_constant_iterator(thrust::nullopt); }
};

}  // namespace cugraph

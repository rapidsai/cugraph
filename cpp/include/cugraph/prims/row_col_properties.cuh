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
#include <cugraph/utilities/thrust_tuple_utils.cuh>

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
class major_properties_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  major_properties_device_view_t() = default;

  major_properties_device_view_t(ValueIterator value_first) : value_first_(value_first) {}

  void add_offset(vertex_t offset) { value_first_ += offset; }

  ValueIterator value_data() const { return value_first_; }

  __device__ ValueIterator get_iter(vertex_t offset) const { return value_first_ + offset; }
  __device__ value_type get(vertex_t offset) const { return *get_iter(offset); }

 private:
  ValueIterator value_first_{};
};

template <typename vertex_t, typename ValueIterator>
class minor_properties_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  minor_properties_device_view_t() = default;

  minor_properties_device_view_t(ValueIterator value_first)
    : key_first_(thrust::nullopt), key_last_(thrust::nullopt), value_first_(value_first)
  {
  }

  minor_properties_device_view_t(vertex_t const* key_first,
                                 vertex_t const* key_last,
                                 ValueIterator value_first)
    : key_first_(key_first), key_last_(key_last), value_first_(value_first)
  {
  }

  __device__ ValueIterator get_iter(vertex_t offset) const
  {
    auto value_offset = offset;
    if (key_first_) {
      auto it = thrust::lower_bound(thrust::seq, *key_first_, *key_last_, offset);
      assert((it != *key_last_) && (*it == offset));
      value_offset = static_cast<vertex_t>(thrust::distance(*key_first_, it));
    }
    return value_first_ + value_offset;
  }

  __device__ value_type get(vertex_t offset) const { return *get_iter(offset); }

 private:
  thrust::optional<vertex_t const*> key_first_{thrust::nullopt};
  thrust::optional<vertex_t const*> key_last_{thrust::nullopt};

  ValueIterator value_first_{};
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
      rmm::exec_policy(stream), value_data(), value_data() + size_dataframe_buffer(buffer_), value);
  }

  auto value_data() { return get_dataframe_buffer_begin(buffer_); }

  auto device_view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);
    return major_properties_device_view_t<vertex_t, decltype(value_first)>(value_first);
  }

  auto mutable_device_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);
    return major_properties_device_view_t<vertex_t, decltype(value_first)>(value_first);
  }

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
      rmm::exec_policy(stream), value_data(), value_data() + size_dataframe_buffer(buffer_), value);
  }

  auto value_data() { return get_dataframe_buffer_begin(buffer_); }

  auto device_view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);
    if (key_first_) {
      return minor_properties_device_view_t<vertex_t, decltype(value_first)>(
        *key_first_, *key_last_, value_first);
    } else {
      return minor_properties_device_view_t<vertex_t, decltype(value_first)>(value_first);
    }
  }

  auto mutable_device_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);
    if (key_first_) {
      return minor_properties_device_view_t<vertex_t, decltype(value_first)>(
        *key_first_, *key_last_, value_first);
    } else {
      return minor_properties_device_view_t<vertex_t, decltype(value_first)>(value_first);
    }
  }

 private:
  std::optional<vertex_t const*> key_first_{std::nullopt};
  std::optional<vertex_t const*> key_last_{std::nullopt};

  decltype(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{})) buffer_;
};

template <typename Iterator,
          typename std::enable_if_t<std::is_arithmetic<
            typename std::iterator_traits<Iterator>::value_type>::value>* = nullptr>
auto to_thrust_tuple(Iterator iter)
{
  return thrust::make_tuple(iter);
}

template <typename Iterator,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<
            typename std::iterator_traits<Iterator>::value_type>::value>* = nullptr>
auto to_thrust_tuple(Iterator iter)
{
  return iter.get_iterator_tuple();
}

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

  row_properties_t() = default;

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

  auto value_data() { return properties_.value_data(); }

  auto device_view() const { return properties_.device_view(); }
  auto mutable_device_view() { return properties_.mutable_device_view(); }

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

  row_properties_t() = default;

  row_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    properties_ = detail::major_properties_t<typename GraphViewType::vertex_type, T>(
      handle, graph_view.get_number_of_local_adj_matrix_partition_rows());
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto value_data() { return properties_.value_data(); }

  auto device_view() const { return properties_.device_view(); }
  auto mutable_device_view() { return properties_.mutable_device_view(); }

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

  col_properties_t() = default;

  col_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    properties_ = detail::major_properties_t<typename GraphViewType::vertex_type, T>(
      handle, graph_view.get_number_of_local_adj_matrix_partition_cols());
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto value_data() { return properties_.value_data(); }

  auto device_view() const { return properties_.device_view(); }
  auto mutable_device_view() { return properties_.mutable_device_view(); }

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

  col_properties_t() = default;

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

  auto value_data() { return properties_.value_data(); }

  auto device_view() const { return properties_.device_view(); }
  auto mutable_device_view() { return properties_.mutable_device_view(); }

 private:
  detail::minor_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

template <typename vertex_t>
class dummy_properties_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  void add_offset(vertex_t offset) {}  // no-op

  __device__ auto get(vertex_t offset) const { return thrust::nullopt; }
};

template <typename vertex_t>
class dummy_properties_t {
 public:
  using value_type = thrust::nullopt_t;

  auto device_view() const { return dummy_properties_device_view_t<vertex_t>{}; }
};

template <typename vertex_t, typename... Ts>
auto device_view_concat(detail::major_properties_device_view_t<vertex_t, Ts>... device_views)
{
  auto concat_first = thrust::make_zip_iterator(
    thrust_tuple_cat(detail::to_thrust_tuple(device_views.value_data())...));
  return detail::major_properties_device_view_t<vertex_t, decltype(concat_first)>(concat_first);
}

}  // namespace cugraph

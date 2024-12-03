/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <type_traits>

namespace cugraph {

namespace detail {

// we cannot use thrust::iterator_traits<Iterator>::value_type if Iterator is void* (reference to
// void is not allowed)
template <typename Iterator, typename Enable = void>
struct optional_dataframe_buffer_iterator_value_type_t;

template <typename Iterator>
struct optional_dataframe_buffer_iterator_value_type_t<
  Iterator,
  std::enable_if_t<!std::is_same_v<Iterator, void*>>> {
  using value = typename thrust::iterator_traits<Iterator>::value_type;
};

template <typename Iterator>
struct optional_dataframe_buffer_iterator_value_type_t<
  Iterator,
  std::enable_if_t<std::is_same_v<Iterator, void*>>> {
  using value = void;
};

template <typename T>
auto allocate_optional_dataframe_buffer(size_t size, rmm::cuda_stream_view stream)
{
  if constexpr (std::is_same_v<T, void>) {
    return std::byte{0};  // dummy
  } else {
    return allocate_dataframe_buffer<T>(size, stream);
  }
}

template <typename T>
struct optional_dataframe_buffer_type {
  using type = decltype(allocate_optional_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}));
};

template <typename T>
using optional_dataframe_buffer_type_t = typename optional_dataframe_buffer_type<T>::type;

template <typename T>
auto get_optional_dataframe_buffer_begin(
  optional_dataframe_buffer_type_t<T>& optional_dataframe_buffer)
{
  if constexpr (std::is_same_v<T, void>) {
    return static_cast<void*>(nullptr);
  } else {
    return get_dataframe_buffer_begin(optional_dataframe_buffer);
  }
}

template <typename T>
auto get_optional_dataframe_buffer_end(
  optional_dataframe_buffer_type_t<T>& optional_dataframe_buffer)
{
  if constexpr (std::is_same_v<T, void>) {
    return static_cast<void*>(nullptr);
  } else {
    return get_dataframe_buffer_end(optional_dataframe_buffer);
  }
}

template <typename T>
auto get_optional_dataframe_buffer_cbegin(
  optional_dataframe_buffer_type_t<T> const& optional_dataframe_buffer)
{
  if constexpr (std::is_same_v<T, void>) {
    return static_cast<void const*>(nullptr);
  } else {
    return get_dataframe_buffer_cbegin(optional_dataframe_buffer);
  }
}

template <typename T>
auto get_optional_dataframe_buffer_cend(
  optional_dataframe_buffer_type_t<T> const& optional_dataframe_buffer)
{
  if constexpr (std::is_same_v<T, void>) {
    return static_cast<void const*>(nullptr);
  } else {
    return get_dataframe_buffer_cend(optional_dataframe_buffer);
  }
}

template <typename T>
void reserve_optional_dataframe_buffer(
  optional_dataframe_buffer_type_t<T>& optional_dataframe_buffer,
  size_t new_buffer_capacity,
  rmm::cuda_stream_view stream_view)
{
  if constexpr (std::is_same_v<T, void>) {
    return;
  } else {
    return reserve_dataframe_buffer(optional_dataframe_buffer, new_buffer_capacity, stream_view);
  }
}

template <typename T>
void resize_optional_dataframe_buffer(
  optional_dataframe_buffer_type_t<T>& optional_dataframe_buffer,
  size_t new_buffer_size,
  rmm::cuda_stream_view stream_view)
{
  if constexpr (std::is_same_v<T, void>) {
    return;
  } else {
    return resize_dataframe_buffer(optional_dataframe_buffer, new_buffer_size, stream_view);
  }
}

template <typename T>
void shrink_to_fit_optional_dataframe_buffer(
  optional_dataframe_buffer_type_t<T>& optional_dataframe_buffer, rmm::cuda_stream_view stream_view)
{
  if constexpr (std::is_same_v<T, void>) {
    return;
  } else {
    return shrink_to_fit_dataframe_buffer(optional_dataframe_buffer, stream_view);
  }
}

template <typename T>
size_t size_optional_dataframe_buffer(
  optional_dataframe_buffer_type_t<T>& optional_dataframe_buffer)
{
  if constexpr (std::is_same_v<T, void>) {
    return size_t{0};
  } else {
    return size_dataframe_buffer(optional_dataframe_buffer);
  }
}

}  // namespace detail

}  // namespace cugraph

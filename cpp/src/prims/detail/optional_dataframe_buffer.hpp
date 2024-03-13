/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
struct optional_dataframe_buffer_value_type_t;

template <typename Iterator>
struct optional_dataframe_buffer_value_type_t<Iterator,
                                              std::enable_if_t<!std::is_same_v<Iterator, void*>>> {
  using value = typename thrust::iterator_traits<Iterator>::value_type;
};

template <typename Iterator>
struct optional_dataframe_buffer_value_type_t<Iterator,
                                              std::enable_if_t<std::is_same_v<Iterator, void*>>> {
  using value = void;
};

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
std::byte allocate_optional_dataframe_buffer(size_t size, rmm::cuda_stream_view stream)
{
  return std::byte{0};  // dummy
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
auto allocate_optional_dataframe_buffer(size_t size, rmm::cuda_stream_view stream)
{
  return allocate_dataframe_buffer<T>(size, stream);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void* get_optional_dataframe_buffer_begin(std::byte& optional_dataframe_buffer)
{
  return static_cast<void*>(nullptr);
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
auto get_optional_dataframe_buffer_begin(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>&
    optional_dataframe_buffer)
{
  return get_dataframe_buffer_begin(optional_dataframe_buffer);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void* get_optional_dataframe_buffer_end(std::byte& optional_dataframe_buffer)
{
  return static_cast<void*>(nullptr);
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
auto get_optional_dataframe_buffer_end(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>&
    optional_dataframe_buffer)
{
  return get_dataframe_buffer_end(optional_dataframe_buffer);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void const* get_optional_dataframe_buffer_cbegin(std::byte const& optional_dataframe_buffer)
{
  return static_cast<void*>(nullptr);
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
auto get_optional_dataframe_buffer_cbegin(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))> const&
    optional_dataframe_buffer)
{
  return get_dataframe_buffer_cbegin(optional_dataframe_buffer);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void const* get_optional_dataframe_buffer_cend(std::byte const& optional_dataframe_buffer)
{
  return static_cast<void*>(nullptr);
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
auto get_optional_dataframe_buffer_cend(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))> const&
    optional_dataframe_buffer)
{
  return get_dataframe_buffer_cend(optional_dataframe_buffer);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void reserve_optional_dataframe_buffer(std::byte& optional_dataframe_buffer,
                                       size_t new_buffer_capacity,
                                       rmm::cuda_stream_view stream_view)
{
  return;
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
void reserve_optional_dataframe_buffer(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>&
    optional_dataframe_buffer,
  size_t new_buffer_capacity,
  rmm::cuda_stream_view stream_view)
{
  return reserve_dataframe_buffer(optional_dataframe_buffer, new_buffer_capacity, stream_view);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void resize_optional_dataframe_buffer(std::byte& optional_dataframe_buffer,
                                      size_t new_buffer_size,
                                      rmm::cuda_stream_view stream_view)
{
  return;
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
void resize_optional_dataframe_buffer(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>&
    optional_dataframe_buffer,
  size_t new_buffer_size,
  rmm::cuda_stream_view stream_view)
{
  return resize_dataframe_buffer(optional_dataframe_buffer, new_buffer_size, stream_view);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
void shrink_to_fit_optional_dataframe_buffer(std::byte& optional_dataframe_buffer,
                                             rmm::cuda_stream_view stream_view)
{
  return;
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
void shrink_to_fit_optional_dataframe_buffer(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>&
    optional_dataframe_buffer,
  rmm::cuda_stream_view stream_view)
{
  return shrink_to_fit_dataframe_buffer(optional_dataframe_buffer, stream_view);
}

template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
size_t size_optional_dataframe_buffer(std::byte const& optional_dataframe_buffer)
{
  return size_t{0};
}

template <typename T, std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
size_t size_optional_dataframe_buffer(
  std::decay_t<decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))> const&
    optional_dataframe_buffer)
{
  return size_dataframe_buffer(optional_dataframe_buffer);
}

}  // namespace detail

}  // namespace cugraph

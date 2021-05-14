/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/utilities/thrust_tuple_utils.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename TupleType, size_t I>
auto allocate_dataframe_buffer_tuple_element_impl(size_t buffer_size, cudaStream_t stream)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return rmm::device_uvector<element_t>(buffer_size, stream);
}

template <typename TupleType, size_t... Is>
auto allocate_dataframe_buffer_tuple_impl(std::index_sequence<Is...>,
                                          size_t buffer_size,
                                          cudaStream_t stream)
{
  return std::make_tuple(
    allocate_dataframe_buffer_tuple_element_impl<TupleType, Is>(buffer_size, stream)...);
}

template <typename TupleType, typename BufferType, size_t I, size_t N>
struct resize_dataframe_buffer_tuple_iterator_element_impl {
  void run(BufferType& buffer, size_t new_buffer_size, cudaStream_t stream)
  {
    std::get<I>(buffer).resize(new_buffer_size, stream);
    resize_dataframe_buffer_tuple_iterator_element_impl<TupleType, BufferType, I + 1, N>().run(
      buffer, new_buffer_size, stream);
  }
};

template <typename TupleType, typename BufferType, size_t I>
struct resize_dataframe_buffer_tuple_iterator_element_impl<TupleType, BufferType, I, I> {
  void run(BufferType& buffer, size_t new_buffer_size, cudaStream_t stream) {}
};

template <typename TupleType, typename BufferType, size_t I, size_t N>
struct shrink_to_fit_dataframe_buffer_tuple_iterator_element_impl {
  void run(BufferType& buffer, cudaStream_t stream)
  {
    std::get<I>(buffer).shrink_to_fit(stream);
    shrink_to_fit_dataframe_buffer_tuple_iterator_element_impl<TupleType, BufferType, I + 1, N>()
      .run(buffer, stream);
  }
};

template <typename TupleType, typename BufferType, size_t I>
struct shrink_to_fit_dataframe_buffer_tuple_iterator_element_impl<TupleType, BufferType, I, I> {
  void run(BufferType& buffer, cudaStream_t stream) {}
};

template <typename TupleType, size_t I, typename BufferType>
auto get_dataframe_buffer_begin_tuple_element_impl(BufferType& buffer)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return std::get<I>(buffer).begin();
}

template <typename TupleType, size_t... Is, typename BufferType>
auto get_dataframe_buffer_begin_tuple_impl(std::index_sequence<Is...>, BufferType& buffer)
{
  // thrust::make_tuple instead of std::make_tuple as this is fed to thrust::make_zip_iterator.
  return thrust::make_tuple(
    get_dataframe_buffer_begin_tuple_element_impl<TupleType, Is>(buffer)...);
}

template <typename TupleType, size_t I, typename BufferType>
auto get_dataframe_buffer_end_tuple_element_impl(BufferType& buffer)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return std::get<I>(buffer).end();
}

template <typename TupleType, size_t... Is, typename BufferType>
auto get_dataframe_buffer_end_tuple_impl(std::index_sequence<Is...>, BufferType& buffer)
{
  // thrust::make_tuple instead of std::make_tuple as this is fed to thrust::make_zip_iterator.
  return thrust::make_tuple(get_dataframe_buffer_end_tuple_element_impl<TupleType, Is>(buffer)...);
}

}  // namespace detail

template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto allocate_dataframe_buffer(size_t buffer_size, cudaStream_t stream)
{
  return rmm::device_uvector<T>(buffer_size, stream);
}

template <typename T, typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto allocate_dataframe_buffer(size_t buffer_size, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return detail::allocate_dataframe_buffer_tuple_impl<T>(
    std::make_index_sequence<tuple_size>(), buffer_size, stream);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
void resize_dataframe_buffer(BufferType& buffer, size_t new_buffer_size, cudaStream_t stream)
{
  buffer.resize(new_buffer_size, stream);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
void resize_dataframe_buffer(BufferType& buffer, size_t new_buffer_size, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  detail::
    resize_dataframe_buffer_tuple_iterator_element_impl<T, BufferType, size_t{0}, tuple_size>()
      .run(buffer, new_buffer_size, stream);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
void shrink_to_fit_dataframe_buffer(BufferType& buffer, cudaStream_t stream)
{
  buffer.shrink_to_fit(stream);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
void shrink_to_fit_dataframe_buffer(BufferType& buffer, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  detail::shrink_to_fit_dataframe_buffer_tuple_iterator_element_impl<T,
                                                                     BufferType,
                                                                     size_t{0},
                                                                     tuple_size>()
    .run(buffer, stream);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
size_t size_dataframe_buffer(BufferType& buffer)
{
  return buffer.size();
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
size_t size_dataframe_buffer(BufferType& buffer)
{
  return std::get<0>(buffer).size();
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto get_dataframe_buffer_begin(BufferType& buffer)
{
  return buffer.begin();
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto get_dataframe_buffer_begin(BufferType& buffer)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return thrust::make_zip_iterator(detail::get_dataframe_buffer_begin_tuple_impl<T>(
    std::make_index_sequence<tuple_size>(), buffer));
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto get_dataframe_buffer_end(BufferType& buffer)
{
  return buffer.end();
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto get_dataframe_buffer_end(BufferType& buffer)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return thrust::make_zip_iterator(
    detail::get_dataframe_buffer_end_tuple_impl<T>(std::make_index_sequence<tuple_size>(), buffer));
}

}  // namespace experimental
}  // namespace cugraph

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

#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {

namespace detail {

template <typename TupleType, size_t... Is>
auto allocate_dataframe_buffer_tuple_impl(std::index_sequence<Is...>,
                                          size_t buffer_size,
                                          rmm::cuda_stream_view stream_view)
{
  return std::make_tuple(rmm::device_uvector<typename thrust::tuple_element<Is, TupleType>::type>(
    buffer_size, stream_view)...);
}

template <typename TupleType, std::size_t... Is>
auto get_dataframe_buffer_begin_tuple_impl(std::index_sequence<Is...>, TupleType& buffer)
{
  return thrust::make_zip_iterator(thrust::make_tuple((std::get<Is>(buffer).begin())...));
}

template <typename TupleType, std::size_t... Is>
auto get_dataframe_buffer_end_tuple_impl(std::index_sequence<Is...>, TupleType& buffer)
{
  return thrust::make_zip_iterator(thrust::make_tuple((std::get<Is>(buffer).end())...));
}

template <typename TupleType, size_t... Is>
auto get_dataframe_buffer_cbegin_tuple_impl(std::index_sequence<Is...>, TupleType& buffer)
{
  // thrust::make_tuple instead of std::make_tuple as this is fed to thrust::make_zip_iterator.
  return thrust::make_zip_iterator(thrust::make_tuple((std::get<Is>(buffer).cbegin())...));
}

template <typename TupleType, std::size_t... Is>
auto get_dataframe_buffer_cend_tuple_impl(std::index_sequence<Is...>, TupleType& buffer)
{
  // thrust::make_tuple instead of std::make_tuple as this is fed to thrust::make_zip_iterator.
  return thrust::make_zip_iterator(thrust::make_tuple((std::get<Is>(buffer).cend())...));
}

}  // namespace detail

template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto allocate_dataframe_buffer(size_t buffer_size, rmm::cuda_stream_view stream_view)
{
  return rmm::device_uvector<T>(buffer_size, stream_view);
}

template <typename T, typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto allocate_dataframe_buffer(size_t buffer_size, rmm::cuda_stream_view stream_view)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return detail::allocate_dataframe_buffer_tuple_impl<T>(
    std::make_index_sequence<tuple_size>(), buffer_size, stream_view);
}

template <typename BufferType>
void reserve_dataframe_buffer(BufferType& buffer,
                              size_t new_buffer_capacity,
                              rmm::cuda_stream_view stream_view)
{
  static_assert(is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value ||
                is_arithmetic_vector<std::remove_cv_t<BufferType>, rmm::device_uvector>::value);
  if constexpr (is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value) {
    std::apply([new_buffer_capacity, stream_view](
                 auto&&... args) { (args.reserve(new_buffer_capacity, stream_view), ...); },
               buffer);
  } else {
    buffer.reserve(new_buffer_capacity, stream_view);
  }
}

template <typename BufferType>
void resize_dataframe_buffer(BufferType& buffer,
                             size_t new_buffer_size,
                             rmm::cuda_stream_view stream_view)
{
  static_assert(is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value ||
                is_arithmetic_vector<std::remove_cv_t<BufferType>, rmm::device_uvector>::value);
  if constexpr (is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value) {
    std::apply([new_buffer_size,
                stream_view](auto&&... args) { (args.resize(new_buffer_size, stream_view), ...); },
               buffer);
  } else {
    buffer.resize(new_buffer_size, stream_view);
  }
}

template <typename BufferType>
void shrink_to_fit_dataframe_buffer(BufferType& buffer, rmm::cuda_stream_view stream_view)
{
  static_assert(is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value ||
                is_arithmetic_vector<std::remove_cv_t<BufferType>, rmm::device_uvector>::value);
  if constexpr (is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value) {
    std::apply([stream_view](auto&&... args) { (args.shrink_to_fit(stream_view), ...); }, buffer);
  } else {
    buffer.shrink_to_fit(stream_view);
  }
}

template <typename BufferType>
size_t size_dataframe_buffer(BufferType& buffer)
{
  static_assert(is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value ||
                is_arithmetic_vector<std::remove_cv_t<BufferType>, rmm::device_uvector>::value);
  if constexpr (is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value) {
    return std::get<0>(buffer).size();
  } else {
    return buffer.size();
  }
}

template <typename BufferType,
          typename std::enable_if_t<is_arithmetic_vector<std::remove_cv_t<BufferType>,
                                                         rmm::device_uvector>::value>* = nullptr>
auto get_dataframe_buffer_begin(BufferType& buffer)
{
  return buffer.begin();
}

template <typename BufferType,
          typename std::enable_if_t<
            is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value>* = nullptr>
auto get_dataframe_buffer_begin(BufferType& buffer)
{
  return detail::get_dataframe_buffer_begin_tuple_impl(
    std::make_index_sequence<std::tuple_size<BufferType>::value>(), buffer);
}

template <typename BufferType,
          typename std::enable_if_t<is_arithmetic_vector<std::remove_cv_t<BufferType>,
                                                         rmm::device_uvector>::value>* = nullptr>
auto get_dataframe_buffer_cbegin(BufferType& buffer)
{
  return buffer.cbegin();
}

template <typename BufferType,
          typename std::enable_if_t<
            is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value>* = nullptr>
auto get_dataframe_buffer_cbegin(BufferType& buffer)
{
  return detail::get_dataframe_buffer_cbegin_tuple_impl(
    std::make_index_sequence<std::tuple_size<BufferType>::value>(), buffer);
}

template <typename BufferType,
          typename std::enable_if_t<is_arithmetic_vector<std::remove_cv_t<BufferType>,
                                                         rmm::device_uvector>::value>* = nullptr>
auto get_dataframe_buffer_end(BufferType& buffer)
{
  return buffer.end();
}

template <typename BufferType,
          typename std::enable_if_t<
            is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value>* = nullptr>
auto get_dataframe_buffer_end(BufferType& buffer)
{
  return detail::get_dataframe_buffer_end_tuple_impl(
    std::make_index_sequence<std::tuple_size<BufferType>::value>(), buffer);
}

template <typename BufferType,
          typename std::enable_if_t<is_arithmetic_vector<std::remove_cv_t<BufferType>,
                                                         rmm::device_uvector>::value>* = nullptr>
auto get_dataframe_buffer_cend(BufferType& buffer)
{
  return buffer.cend();
}

template <typename BufferType,
          typename std::enable_if_t<
            is_std_tuple_of_arithmetic_vectors<std::remove_cv_t<BufferType>>::value>* = nullptr>
auto get_dataframe_buffer_cend(BufferType& buffer)
{
  return detail::get_dataframe_buffer_cend_tuple_impl(
    std::make_index_sequence<std::tuple_size<BufferType>::value>(), buffer);
}

template <typename T>
struct dataframe_buffer_value_type {
  using type = void;
};

template <typename T>
struct dataframe_buffer_value_type<rmm::device_uvector<T>> {
  using type = T;
};

template <typename... Ts>
struct dataframe_buffer_value_type<std::tuple<rmm::device_uvector<Ts>...>> {
  using type = thrust::tuple<Ts...>;
};

template <typename BufferType>
using dataframe_buffer_value_type_t = typename dataframe_buffer_value_type<BufferType>::type;

template <typename T>
struct dataframe_buffer_type {
  using type = decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}));
};

template <typename T>
using dataframe_buffer_type_t = typename dataframe_buffer_type<T>::type;

}  // namespace cugraph

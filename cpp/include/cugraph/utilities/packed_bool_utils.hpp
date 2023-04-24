/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

template <typename ValueIterator, typename value_t, std::size_t... Is>
constexpr std::enable_if_t<cugraph::is_thrust_tuple_of_arithmetic<
                             typename thrust::iterator_traits<ValueIterator>::value_type>::value &&
                             cugraph::is_thrust_tuple_of_arithmetic<value_t>::value,
                           bool>
has_packed_bool_element(std::index_sequence<Is...>)
{
  static_assert(
    thrust::tuple_size<typename thrust::iterator_traits<ValueIterator>::value_type>::value ==
    thrust::tuple_size<value_t>::value);
  return (... ||
          (std::is_same_v<typename thrust::tuple_element<
                            Is,
                            typename thrust::iterator_traits<ValueIterator>::value_type>::type,
                          uint32_t> &&
           std::is_same_v<typename thrust::tuple_element<Is, value_t>::type, bool>));
}

}  // namespace detail

// sizeof(uint32_t) * 8 packed Boolean values are stored using one uint32_t
template <typename ValueIterator, typename value_t>
constexpr bool has_packed_bool_element()
{
  static_assert(
    (std::is_arithmetic_v<typename thrust::iterator_traits<ValueIterator>::value_type> &&
     std::is_arithmetic_v<value_t>) ||
    (cugraph::is_thrust_tuple_of_arithmetic<
       typename thrust::iterator_traits<ValueIterator>::value_type>::value &&
     cugraph::is_thrust_tuple_of_arithmetic<value_t>::value));
  if constexpr (std::is_arithmetic_v<typename thrust::iterator_traits<ValueIterator>::value_type> &&
                std::is_arithmetic_v<value_t>) {
    return std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, uint32_t> &&
           std::is_same_v<value_t, bool>;
  } else {
    static_assert(
      thrust::tuple_size<typename thrust::iterator_traits<ValueIterator>::value_type>::value ==
      thrust::tuple_size<value_t>::value);
    return detail::has_packed_bool_element<ValueIterator, value_t>(
      std::make_index_sequence<thrust::tuple_size<value_t>::value>());
  }
}

constexpr size_t packed_bools_per_word() { return sizeof(uint32_t) * size_t{8}; }

constexpr size_t packed_bool_size(size_t bool_size)
{
  return (bool_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8);
}

template <typename T>
constexpr uint32_t packed_bool_mask(T bool_offset)
{
  return uint32_t{1} << (bool_offset % (sizeof(uint32_t) * 8));
}

constexpr uint32_t packed_bool_full_mask() { return uint32_t{0xffffffff}; }

constexpr uint32_t packed_bool_empty_mask() { return uint32_t{0x0}; }

template <typename T>
constexpr T packed_bool_offset(T bool_offset)
{
  return bool_offset / (sizeof(uint32_t) * 8);
}

}  // namespace cugraph

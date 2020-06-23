/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <array>
#include <type_traits>

namespace {

template <typename TupleType, size_t I, size_t N>
struct is_thrust_tuple_of_arithemetic_impl {
  constexpr bool evaluate() const
  {
    if (!std::is_arithmetic<typename thrust::tuple_element<I, TupleType>::type>::value) {
      return false;
    } else {
      return is_thrust_tuple_of_arithemetic_impl<TupleType, I + 1, N>().evaluate();
    }
  }
};

template <typename TupleType, size_t I>
struct is_thrust_tuple_of_arithemetic_impl<TupleType, I, I> {
  constexpr bool evaluate() const { return true; }
};

template <typename TupleType, size_t I, size_t N>
struct compute_thrust_tuple_element_sizes_impl {
  void compute(std::array<size_t, thrust::tuple_size<TupleType>::value>& arr) const
  {
    arr[I] = sizeof(typename thrust::tuple_element<I, TupleType>::type);
    compute_thrust_tuple_element_sizes_impl<TupleType, I + 1, N>().compute(arr);
  }
};

template <typename TupleType, size_t I>
struct compute_thrust_tuple_element_sizes_impl<TupleType, I, I> {
  void compute(std::array<size_t, thrust::tuple_size<TupleType>::value>& arr) const {}
};

template <typename TupleType, size_t... Is>
__device__ constexpr auto remove_first_thrust_tuple_element_impl(TupleType const& tuple,
                                                                 std::index_sequence<Is...>)
{
  return thrust::make_tuple(thrust::get<1 + Is>(tuple)...);
}

template <typename TupleType, size_t I, size_t N>
struct plus_thrust_tuple_impl {
  __device__ constexpr void compute(TupleType& lhs, TupleType const& rhs) const
  {
    thrust::get<I>(lhs) += thrust::get<I>(rhs);
    plus_thrust_tuple_impl<TupleType, I + 1, N>().compute(lhs, rhs);
  }
};

template <typename TupleType, size_t I>
struct plus_thrust_tuple_impl<TupleType, I, I> {
  __device__ constexpr void compute(TupleType& lhs, TupleType const& rhs) const {}
};

template <typename TupleType, size_t BlockSize, size_t I, size_t N>
struct block_reduce_thrust_tuple_impl {
  __device__ void compute(TupleType& tuple) const
  {
    using T           = typename thrust::tuple_element<I, TupleType>::type;
    using BlockReduce = cub::BlockReduce<T, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    thrust::get<I>(tuple) = BlockReduce(temp_storage).Sum(thrust::get<I>(tuple));
  }
};

template <typename TupleType, size_t BlockSize, size_t I>
struct block_reduce_thrust_tuple_impl<TupleType, BlockSize, I, I> {
  __device__ void compute(TupleType& tuple) const {}
};

}  // namespace

namespace cugraph {
namespace experimental {
namespace detail {

template <typename T>
struct is_thrust_tuple : std::false_type {
};

template <typename... Ts>
struct is_thrust_tuple<thrust::tuple<Ts...>> : std::true_type {
};

template <typename TupleType, typename Enable = void>
struct is_thrust_tuple_of_arithmetic : std::false_type {
};

template <typename TupleType>
struct is_thrust_tuple_of_arithmetic<TupleType,
                                     std::enable_if_t<is_thrust_tuple<TupleType>::value>> {
  static constexpr bool value =
    is_thrust_tuple_of_arithemetic_impl<TupleType,
                                        static_cast<size_t>(0),
                                        static_cast<size_t>(thrust::tuple_size<TupleType>::value)>()
      .evaluate();
};

template <typename T>
struct is_arithmetic_or_thrust_tuple_of_arithmetic
  : std::integral_constant<bool, std::is_arithmetic<T>::value> {
};

template <typename... Ts>
struct is_arithmetic_or_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>>
  : std::integral_constant<bool, is_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>>::value> {
};

template <typename TupleType>
struct compute_thrust_tuple_element_sizes {
  auto operator()() const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    std::array<size_t, tuple_size> ret;
    compute_thrust_tuple_element_sizes_impl<TupleType, static_cast<size_t>(0), tuple_size>()
      .compute(ret);
    return ret;
  }
};

template <typename TupleType>
struct remove_first_thrust_tuple_element {
  __device__ constexpr auto operator()(TupleType const& tuple) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    return remove_first_thrust_tuple_element_impl(tuple,
                                                  std::make_index_sequence<tuple_size - 1>());
  }
};

template <typename TupleType>
struct plus_thrust_tuple {
  __device__ constexpr TupleType operator()(TupleType const& lhs, TupleType const& rhs) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    auto ret                    = lhs;
    plus_thrust_tuple_impl<TupleType, static_cast<size_t>(0), tuple_size>().compute(ret, rhs);
    return ret;
  }
};

template <typename TupleType, size_t BlockSize>
struct block_reduce_thrust_tuple {
  __device__ TupleType operator()(TupleType const& tuple) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    auto ret                    = tuple;
    block_reduce_thrust_tuple_impl<TupleType, BlockSize, static_cast<size_t>(0), tuple_size>()
      .compute(ret);
    return ret;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
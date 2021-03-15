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

#include <raft/cudart_utils.h>
#include <raft/device_atomics.cuh>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <array>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

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
  __host__ __device__ constexpr void compute(TupleType& lhs, TupleType const& rhs) const
  {
    thrust::get<I>(lhs) += thrust::get<I>(rhs);
    plus_thrust_tuple_impl<TupleType, I + 1, N>().compute(lhs, rhs);
  }
};

template <typename TupleType, size_t I>
struct plus_thrust_tuple_impl<TupleType, I, I> {
  __host__ __device__ constexpr void compute(TupleType& lhs, TupleType const& rhs) const {}
};

template <typename T>
__device__ std::enable_if_t<std::is_arithmetic<T>::value, void> atomic_accumulate_impl(
  thrust::detail::any_assign& /* dereferencing thrust::discard_iterator results in this type */ lhs,
  T const& rhs)
{
  // no-op
}

template <typename T>
__device__ std::enable_if_t<std::is_arithmetic<T>::value, void> atomic_accumulate_impl(T& lhs,
                                                                                       T const& rhs)
{
  atomicAdd(&lhs, rhs);
}

template <typename Iterator, typename TupleType, size_t I, size_t N>
struct atomic_accumulate_thrust_tuple_impl {
  __device__ constexpr void compute(Iterator iter, TupleType const& value) const
  {
    atomic_accumulate_impl(thrust::raw_reference_cast(thrust::get<I>(*iter)),
                           thrust::get<I>(value));
    atomic_accumulate_thrust_tuple_impl<Iterator, TupleType, I + 1, N>().compute(iter, value);
  }
};

template <typename Iterator, typename TupleType, size_t I>
struct atomic_accumulate_thrust_tuple_impl<Iterator, TupleType, I, I> {
  __device__ constexpr void compute(Iterator iter, TupleType const& value) const {}
};

template <typename TupleType, size_t I, size_t N>
struct warp_reduce_thrust_tuple_impl {
  __device__ void compute(TupleType& tuple) const
  {
    auto& val = thrust::get<I>(tuple);
    for (auto offset = raft::warp_size() / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(raft::warp_full_mask(), val, offset);
    }
  }
};

template <typename TupleType, size_t I>
struct warp_reduce_thrust_tuple_impl<TupleType, I, I> {
  __device__ void compute(TupleType& tuple) const {}
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

}  // namespace detail

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
    detail::is_thrust_tuple_of_arithemetic_impl<TupleType,
                                                0,
                                                static_cast<size_t>(
                                                  thrust::tuple_size<TupleType>::value)>()
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

template <typename T>
struct thrust_tuple_size_or_one : std::integral_constant<size_t, 1> {
};

template <typename... Ts>
struct thrust_tuple_size_or_one<thrust::tuple<Ts...>>
  : std::integral_constant<size_t, thrust::tuple_size<thrust::tuple<Ts...>>::value> {
};

template <typename TupleType>
struct compute_thrust_tuple_element_sizes {
  auto operator()() const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    std::array<size_t, tuple_size> ret;
    detail::compute_thrust_tuple_element_sizes_impl<TupleType, size_t{0}, tuple_size>().compute(
      ret);
    return ret;
  }
};

template <typename TupleType>
struct remove_first_thrust_tuple_element {
  __device__ constexpr auto operator()(TupleType const& tuple) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    return detail::remove_first_thrust_tuple_element_impl(
      tuple, std::make_index_sequence<tuple_size - 1>());
  }
};

template <typename TupleType>
struct plus_thrust_tuple {
  __host__ __device__ constexpr TupleType operator()(TupleType const& lhs,
                                                     TupleType const& rhs) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    auto ret                    = lhs;
    detail::plus_thrust_tuple_impl<TupleType, size_t{0}, tuple_size>().compute(ret, rhs);
    return ret;
  }
};

template <typename Iterator, typename TupleType>
struct atomic_accumulate_thrust_tuple {
  __device__ constexpr void operator()(Iterator iter, TupleType const& value) const
  {
    static_assert(
      thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
      thrust::tuple_size<TupleType>::value);
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    detail::atomic_accumulate_thrust_tuple_impl<Iterator, TupleType, size_t{0}, tuple_size>()
      .compute(iter, value);
  }
};

template <typename TupleType>
struct warp_reduce_thrust_tuple {  // only warp lane 0 has a valid result
  __device__ TupleType operator()(TupleType const& tuple) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    auto ret                    = tuple;
    detail::warp_reduce_thrust_tuple_impl<TupleType, size_t{0}, tuple_size>().compute(ret);
    return ret;
  }
};

template <typename TupleType, size_t BlockSize>
struct block_reduce_thrust_tuple {
  __device__ TupleType operator()(TupleType const& tuple) const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    auto ret                    = tuple;
    detail::block_reduce_thrust_tuple_impl<TupleType, BlockSize, size_t{0}, tuple_size>().compute(
      ret);
    return ret;
  }
};

}  // namespace experimental
}  // namespace cugraph

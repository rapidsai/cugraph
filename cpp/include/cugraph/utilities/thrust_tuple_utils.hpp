/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

#include <array>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename TupleType, size_t I, size_t N>
struct is_thrust_tuple_of_arithemetic_impl {
  constexpr bool evaluate() const
  {
    if (!std::is_arithmetic_v<typename thrust::tuple_element<I, TupleType>::type>) {
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

template <typename TupleType, std::size_t... Is>
size_t sum_thrust_tuple_element_sizes(std::index_sequence<Is...>)
{
  return (... + sizeof(typename thrust::tuple_element<Is, TupleType>::type));
}

template <typename TupleType, std::size_t... Is>
size_t min_thrust_tuple_element_sizes(std::index_sequence<Is...>)
{
  return std::min(sizeof(typename thrust::tuple_element<Is, TupleType>::type)...);
}

template <typename TupleType, std::size_t... Is>
size_t max_thrust_tuple_element_sizes(std::index_sequence<Is...>)
{
  return std::max(sizeof(typename thrust::tuple_element<Is, TupleType>::type)...);
}

template <typename TupleType, std::size_t... Is>
auto thrust_tuple_to_std_tuple(TupleType tup, std::index_sequence<Is...>)
{
  return std::make_tuple(thrust::get<Is>(tup)...);
}

template <typename TupleType, std::size_t... Is>
auto std_tuple_to_thrust_tuple(TupleType tup, std::index_sequence<Is...>)
{
  constexpr size_t maximum_thrust_tuple_size = 10;
  static_assert(std::tuple_size_v<TupleType> <= maximum_thrust_tuple_size);
  return thrust::make_tuple(std::get<Is>(tup)...);
}

template <typename... Ts, typename... Us, std::size_t... I0, std::size_t... I1>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_cat_impl(thrust::tuple<Ts...> tup0,
                        thrust::tuple<Us...> tup1,
                        std::index_sequence<I0...>,
                        std::index_sequence<I1...>)
{
  return thrust::make_tuple(thrust::get<I0>(tup0)..., thrust::get<I1>(tup1)...);
}

template <typename... Ts, typename... Us>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_cat_impl(thrust::tuple<Ts...> tup0, thrust::tuple<Us...> tup1)
{
  return thrust_tuple_cat_impl(
    tup0, tup1, std::index_sequence_for<Ts...>{}, std::index_sequence_for<Us...>{});
}

template <typename TupleType>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_cat_impl(TupleType tup)
{
  return tup;
}

template <typename TupleType0, typename TupleType1, typename... RemainingTupleTypes>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_cat_impl(TupleType0 tup0, TupleType1 tup1, RemainingTupleTypes... tups)
{
  return thrust_tuple_cat_impl(thrust_tuple_cat_impl(tup0, tup1), tups...);
}

template <typename TupleType, size_t F, size_t... Is>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_slice_impl(TupleType tup, std::index_sequence<Is...>)
{
  return thrust::make_tuple(thrust::get<F + Is>(tup)...);
}

template <typename TupleType, std::size_t... Is>
constexpr TupleType thrust_tuple_of_arithmetic_numeric_limits_lowest(std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    std::numeric_limits<typename thrust::tuple_element<Is, TupleType>::type>::lowest()...);
}

template <typename TupleType, std::size_t... Is>
constexpr TupleType thrust_tuple_of_arithmetic_numeric_limits_max(std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    std::numeric_limits<typename thrust::tuple_element<Is, TupleType>::type>::max()...);
}

}  // namespace detail

template <typename T>
struct is_thrust_tuple : std::false_type {};

template <typename... Ts>
struct is_thrust_tuple<thrust::tuple<Ts...>> : std::true_type {};

template <typename T>
constexpr bool is_thrust_tuple_v = is_thrust_tuple<T>::value;

template <typename T>
struct is_thrust_tuple_of_arithmetic : std::false_type {};

template <typename... Ts>
struct is_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>> {
 private:
  template <typename T>
  static constexpr bool is_valid = std::is_arithmetic_v<T>;

 public:
  static constexpr bool value = (... && is_valid<Ts>);
};

template <typename T>
constexpr bool is_thrust_tuple_of_arithmetic_v = is_thrust_tuple_of_arithmetic<T>::value;

template <typename T>
struct is_thrust_tuple_of_integral : std::false_type {};

template <typename... Ts>
struct is_thrust_tuple_of_integral<thrust::tuple<Ts...>> {
 private:
  template <typename T>
  static constexpr bool is_valid = std::is_integral_v<T>;

 public:
  static constexpr bool value = (... && is_valid<Ts>);
};

template <typename T>
constexpr bool is_thrust_tuple_of_integral_v = is_thrust_tuple_of_integral<T>::value;

template <typename T>
struct is_std_tuple : std::false_type {};

template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T, template <typename> typename Vector>
struct is_arithmetic_vector : std::false_type {};

template <template <typename> typename Vector, typename T>
struct is_arithmetic_vector<Vector<T>, Vector>
  : std::integral_constant<bool, std::is_arithmetic_v<T>> {};

template <typename T>
struct is_std_tuple_of_arithmetic_vectors : std::false_type {};

template <typename... Ts>
struct is_std_tuple_of_arithmetic_vectors<std::tuple<rmm::device_uvector<Ts>...>> {
  static constexpr bool value = (... && std::is_arithmetic_v<Ts>);
};

template <typename T>
struct is_arithmetic_or_thrust_tuple_of_arithmetic
  : std::integral_constant<bool, std::is_arithmetic_v<T>> {};

template <typename... Ts>
struct is_arithmetic_or_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>>
  : std::integral_constant<bool, is_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>>::value> {};

template <typename T>
constexpr bool is_arithmetic_or_thrust_tuple_of_arithmetic_v =
  is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value;

template <typename T>
struct thrust_tuple_size_or_one : std::integral_constant<size_t, 1> {};

template <typename... Ts>
struct thrust_tuple_size_or_one<thrust::tuple<Ts...>>
  : std::integral_constant<size_t, thrust::tuple_size<thrust::tuple<Ts...>>::value> {};

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
constexpr size_t sum_thrust_tuple_element_sizes()
{
  return detail::sum_thrust_tuple_element_sizes<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename TupleType>
constexpr size_t min_thrust_tuple_element_sizes()
{
  return detail::min_thrust_tuple_element_sizes<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename TupleType>
constexpr size_t max_thrust_tuple_element_sizes()
{
  return detail::max_thrust_tuple_element_sizes<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename TupleType>
auto thrust_tuple_to_std_tuple(TupleType tup)
{
  return detail::thrust_tuple_to_std_tuple(
    tup, std::make_index_sequence<thrust::tuple_size<TupleType>::value>{});
}

template <typename TupleType>
auto std_tuple_to_thrust_tuple(TupleType tup)
{
  constexpr size_t maximum_thrust_tuple_size = 10;
  static_assert(std::tuple_size_v<TupleType> <= maximum_thrust_tuple_size);
  return detail::std_tuple_to_thrust_tuple(
    tup, std::make_index_sequence<std::tuple_size_v<TupleType>>{});
}

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  to_thrust_tuple(T scalar_value)
{
  return thrust::make_tuple(scalar_value);
}

template <typename... Ts>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  to_thrust_tuple(thrust::tuple<Ts...> tuple_value)
{
  return tuple_value;
}

template <typename Iterator,
          typename std::enable_if_t<
            std::is_arithmetic_v<typename std::iterator_traits<Iterator>::value_type>>* = nullptr>
auto to_thrust_iterator_tuple(Iterator iter)
{
  return thrust::make_tuple(iter);
}

template <typename Iterator,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<
            typename std::iterator_traits<Iterator>::value_type>::value>* = nullptr>
auto to_thrust_iterator_tuple(Iterator iter)
{
  return iter.get_iterator_tuple();
}

template <typename T, size_t I, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_get_or_identity(T val)
{
  return val;
}

template <typename T,
          size_t I,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_get_or_identity(T val)
{
  return thrust::get<I>(val);
}

template <typename Iterator,
          size_t I,
          typename std::enable_if_t<std::is_arithmetic_v<
            typename thrust::iterator_traits<Iterator>::value_type>>* = nullptr>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_get_or_identity(Iterator val)
{
  return val;
}

template <typename Iterator,
          size_t I,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<
            typename thrust::iterator_traits<Iterator>::value_type>::value>* = nullptr>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_get_or_identity(Iterator val)
{
  return thrust::get<I>(val.get_iterator_tuple());
}

// a temporary function to emulate thrust::tuple_cat (should retire once thrust::tuple is replaced
// with cuda::std::tuple)
template <typename... TupleTypes>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_cat(TupleTypes... tups)
{
  return detail::thrust_tuple_cat_impl(tups...);
}

template <typename TupleType, size_t F /* first (inclusive) */, size_t L /* last (exclusive) */>
#ifdef __CUDACC__
__host__ __device__
#endif
  auto
  thrust_tuple_slice(TupleType tup)
{
  static_assert(L > F);
  return detail::thrust_tuple_slice_impl<TupleType, F>(tup, std::make_index_sequence<L - F>());
}

template <typename TupleType>
constexpr TupleType thrust_tuple_of_arithmetic_numeric_limits_lowest()
{
  return detail::thrust_tuple_of_arithmetic_numeric_limits_lowest<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename TupleType>
constexpr TupleType thrust_tuple_of_arithmetic_numeric_limits_max()
{
  return detail::thrust_tuple_of_arithmetic_numeric_limits_max<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename TupleType, size_t I>
struct thrust_tuple_get {
  __device__ typename thrust::tuple_element<I, TupleType>::type operator()(TupleType tup) const
  {
    return thrust::get<I>(tup);
  }
};

template <typename T, typename... Ts>
decltype(auto) get_first_of_pack(T&& t, Ts&&...)
{
  return std::forward<T>(t);
}

}  // namespace cugraph

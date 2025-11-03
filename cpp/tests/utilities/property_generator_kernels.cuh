/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/property_generator_utilities.hpp"

#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <cuco/hash_functions.cuh>

#include <type_traits>
#include <utility>

namespace cugraph {
namespace test {

namespace detail {

template <typename TupleType, typename T, std::size_t... Is>
__host__ __device__ auto make_type_casted_tuple_from_scalar(T val, std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    static_cast<typename cuda::std::tuple_element<Is, TupleType>::type>(val)...);
}

template <typename property_t, typename T>
__host__ __device__ auto make_property_value(T val)
{
  property_t ret{};
  if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
    ret = make_type_casted_tuple_from_scalar<property_t>(
      val, std::make_index_sequence<cuda::std::tuple_size<property_t>::value>{});
  } else {
    ret = static_cast<property_t>(val);
  }
  return ret;
}

template <typename vertex_t, typename property_t>
struct vertex_property_transform {
  int32_t mod{};

  constexpr __device__ property_t operator()(vertex_t v) const
  {
    static_assert(cugraph::is_thrust_tuple_of_arithmetic<property_t>::value ||
                  std::is_arithmetic_v<property_t>);
    cuco::murmurhash3_32<vertex_t> hash_func{};
    return make_property_value<property_t>(hash_func(v) % mod);
  }
};

template <typename vertex_t, typename property_t>
struct edge_property_transform {
  int32_t mod{};

  constexpr __device__ property_t operator()(vertex_t src,
                                             vertex_t dst,
                                             cuda::std::nullopt_t,
                                             cuda::std::nullopt_t,
                                             cuda::std::nullopt_t) const
  {
    static_assert(cugraph::is_thrust_tuple_of_arithmetic<property_t>::value ||
                  std::is_arithmetic_v<property_t>);
    cuco::murmurhash3_32<vertex_t> hash_func{};
    return make_property_value<property_t>(hash_func(src + dst) % mod);
  }
};

}  // namespace detail

}  // namespace test
}  // namespace cugraph

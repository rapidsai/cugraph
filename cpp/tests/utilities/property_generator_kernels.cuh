/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/property_generator_utilities.hpp"

#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <cuco/hash_functions.cuh>

#include <type_traits>
#include <utility>

namespace cugraph {
namespace test {

namespace detail {

template <typename TupleType, typename T, std::size_t... Is>
__host__ __device__ auto make_type_casted_tuple_from_scalar(T val, std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    static_cast<typename thrust::tuple_element<Is, TupleType>::type>(val)...);
}

template <typename property_t, typename T>
__host__ __device__ auto make_property_value(T val)
{
  property_t ret{};
  if constexpr (cugraph::is_thrust_tuple_of_arithmetic<property_t>::value) {
    ret = make_type_casted_tuple_from_scalar<property_t>(
      val, std::make_index_sequence<thrust::tuple_size<property_t>::value>{});
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

  constexpr __device__ property_t operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
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

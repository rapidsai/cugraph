/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <thrust/equal.h>
#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace cugraph {
namespace test {

namespace detail {

template <typename T>
__host__ __device__ bool compare_arithmetic_scalar(T val0,
                                                   T val1,
                                                   thrust::optional<T> threshold_ratio)
{
  if (threshold_ratio) {
    return std::abs(val0 - val1) <= (std::max(std::abs(val0), std::abs(val1)) * *threshold_ratio);
  } else {
    return val0 == val1;
  }
}

}  // namespace detail

template <typename T>
struct comparator {
  static constexpr double threshold_ratio{1e-2};

  __host__ __device__ bool operator()(T t0, T t1) const
  {
    static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
    if constexpr (std::is_arithmetic_v<T>) {
      return detail::compare_arithmetic_scalar(
        t0,
        t1,
        std::is_floating_point_v<T> ? thrust::optional<T>{threshold_ratio} : thrust::nullopt);
    } else {
      auto val0   = thrust::get<0>(t0);
      auto val1   = thrust::get<0>(t1);
      auto passed = detail::compare_arithmetic_scalar(
        val0,
        val1,
        std::is_floating_point_v<decltype(val0)> ? thrust::optional<decltype(val0)>{threshold_ratio}
                                                 : thrust::nullopt);
      if (!passed) return false;

      if constexpr (thrust::tuple_size<T>::value >= 2) {
        auto val0 = thrust::get<1>(t0);
        auto val1 = thrust::get<1>(t1);
        auto passed =
          detail::compare_arithmetic_scalar(val0,
                                            val1,
                                            std::is_floating_point_v<decltype(val1)>
                                              ? thrust::optional<decltype(val1)>{threshold_ratio}
                                              : thrust::nullopt);
        if (!passed) return false;
      }
      if constexpr (thrust::tuple_size<T>::value >= 3) {
        assert(false);  // should not be reached.
      }
      return true;
    }
  }
};

struct scalar_result_compare {
  template <typename... Args>
  auto operator()(thrust::tuple<Args...> t1, thrust::tuple<Args...> t2)
  {
    using type = thrust::tuple<Args...>;
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<type>::value>());
  }

  template <typename T>
  auto operator()(T t1, T t2)
  {
    comparator<T> comp{};
    return comp(t1, t2);
  }

 private:
  template <typename T, std::size_t... I>
  auto equality_impl(T t1, T t2, std::index_sequence<I...>)
  {
    return (... && (scalar_result_compare::operator()(thrust::get<I>(t1), thrust::get<I>(t2))));
  }
};

struct vector_result_compare {
  const raft::handle_t& handle_;

  vector_result_compare(raft::handle_t const& handle) : handle_(handle) {}

  template <typename... Args>
  auto operator()(std::tuple<rmm::device_uvector<Args>...> const& t1,
                  std::tuple<rmm::device_uvector<Args>...> const& t2)
  {
    using type = thrust::tuple<Args...>;
    return equality_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<type>::value>());
  }

  template <typename T>
  auto operator()(rmm::device_uvector<T> const& t1, rmm::device_uvector<T> const& t2)
  {
    return thrust::equal(
      handle_.get_thrust_policy(), t1.begin(), t1.end(), t2.begin(), comparator<T>());
  }

 private:
  template <typename T, std::size_t... I>
  auto equality_impl(T& t1, T& t2, std::index_sequence<I...>)
  {
    return (... && (vector_result_compare::operator()(std::get<I>(t1), std::get<I>(t2))));
  }
};

}  // namespace test
}  // namespace cugraph

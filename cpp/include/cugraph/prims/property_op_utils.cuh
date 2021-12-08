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

#include <raft/comms/comms.hpp>
#include <raft/device_atomics.cuh>

#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <array>
#include <type_traits>

namespace cugraph {

template <typename InvokeResultEdgeOp, typename Enable = void>
struct is_valid_edge_op {
  static constexpr bool value = false;
};

template <typename InvokeResultEdgeOp>
struct is_valid_edge_op<
  InvokeResultEdgeOp,
  typename std::conditional_t<false, typename InvokeResultEdgeOp::type, void>> {
  static constexpr bool valid = true;
};

template <typename GraphViewType,
          typename key_t,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp>
struct evaluate_edge_op {
  using vertex_type    = typename GraphViewType::vertex_type;
  using weight_type    = typename GraphViewType::weight_type;
  using row_value_type = typename AdjMatrixRowValueInputWrapper::value_type;
  using col_value_type = typename AdjMatrixColValueInputWrapper::value_type;

  template <typename K = key_t,
            typename V = vertex_type,
            typename W = weight_type,
            typename R = row_value_type,
            typename C = col_value_type,
            typename E = EdgeOp>
  __device__
    std::enable_if_t<is_valid_edge_op<typename std::invoke_result<E, K, V, W, R, C>>::valid,
                     typename std::invoke_result<E, K, V, W, R, C>::type>
    compute(K r, V c, W w, R rv, C cv, E e)
  {
    return e(r, c, w, rv, cv);
  }

  template <typename K = key_t,
            typename V = vertex_type,
            typename W = weight_type,
            typename R = row_value_type,
            typename C = col_value_type,
            typename E = EdgeOp>
  __device__ std::enable_if_t<is_valid_edge_op<typename std::invoke_result<E, K, V, R, C>>::valid,
                              typename std::invoke_result<E, K, V, R, C>::type>
  compute(K r, V c, W w, R rv, C cv, E e)
  {
    return e(r, c, rv, cv);
  }
};

template <typename GraphViewType,
          typename key_t,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename T>
struct cast_edge_op_bool_to_integer {
  static_assert(std::is_integral<T>::value);
  using vertex_type    = typename GraphViewType::vertex_type;
  using weight_type    = typename GraphViewType::weight_type;
  using row_value_type = typename AdjMatrixRowValueInputWrapper::value_type;
  using col_value_type = typename AdjMatrixColValueInputWrapper::value_type;

  EdgeOp e_op{};

  template <typename K = key_t,
            typename V = vertex_type,
            typename W = weight_type,
            typename R = row_value_type,
            typename C = col_value_type,
            typename E = EdgeOp>
  __device__
    std::enable_if_t<is_valid_edge_op<typename std::invoke_result<E, K, V, W, R, C>>::valid, T>
    operator()(K r, V c, W w, R rv, C cv)
  {
    return e_op(r, c, w, rv, cv) ? T{1} : T{0};
  }

  template <typename K = key_t,
            typename V = vertex_type,
            typename R = row_value_type,
            typename C = col_value_type,
            typename E = EdgeOp>
  __device__
    std::enable_if_t<is_valid_edge_op<typename std::invoke_result<E, K, V, R, C>>::valid, T>
    operator()(K r, V c, R rv, C cv)
  {
    return e_op(r, c, rv, cv) ? T{1} : T{0};
  }
};

template <typename T, template <typename> typename Op>
struct property_op : public Op<T> {
};

template <typename... Args, template <typename> typename Op>
struct property_op<thrust::tuple<Args...>, Op>
  : public thrust::
      binary_function<thrust::tuple<Args...>, thrust::tuple<Args...>, thrust::tuple<Args...>> {
  using Type = thrust::tuple<Args...>;

 private:
  template <typename T, std::size_t... Is>
  __host__ __device__ constexpr auto sum_impl(T& t1, T& t2, std::index_sequence<Is...>)
  {
    return thrust::make_tuple((Op<typename thrust::tuple_element<Is, Type>::type>()(
      thrust::get<Is>(t1), thrust::get<Is>(t2)))...);
  }

 public:
  __host__ __device__ constexpr auto operator()(const Type& t1, const Type& t2)
  {
    return sum_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<Type>::value>());
  }
};

template <typename T, typename F>
constexpr auto op_dispatch(raft::comms::op_t op, F&& f)
{
  switch (op) {
    case raft::comms::op_t::SUM: {
      return std::invoke(f, property_op<T, thrust::plus>());
    } break;
    case raft::comms::op_t::MIN: {
      return std::invoke(f, property_op<T, thrust::minimum>());
    } break;
    case raft::comms::op_t::MAX: {
      return std::invoke(f, property_op<T, thrust::maximum>());
    } break;
    default: {
      CUGRAPH_FAIL("Unhandled raft::comms::op_t");
      return std::invoke_result_t<F, property_op<T, thrust::plus>>{};
    }
  };
}

template <typename T>
T identity_element(raft::comms::op_t op)
{
  switch (op) {
    case raft::comms::op_t::SUM: {
      return T{0};
    } break;
    case raft::comms::op_t::MIN: {
      return std::numeric_limits<T>::max();
    } break;
    case raft::comms::op_t::MAX: {
      return std::numeric_limits<T>::lowest();
    } break;
    default: {
      CUGRAPH_FAIL("Unhandled raft::comms::op_t");
      return T{0};
    }
  };
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void>
atomic_accumulate_edge_op_result(Iterator iter, T const& value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_same<typename thrust::iterator_traits<Iterator>::value_type, T>::value &&
                     std::is_arithmetic<T>::value,
                   void>
  atomic_accumulate_edge_op_result(Iterator iter, T const& value)
{
  atomicAdd(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   void>
  atomic_accumulate_edge_op_result(Iterator iter, T const& value)
{
  static_assert(thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
                thrust::tuple_size<T>::value);
  atomic_accumulate_thrust_tuple<Iterator, T>()(iter, value);
  return;
}

}  // namespace cugraph

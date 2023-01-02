/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <prims/property_op_utils.cuh>

#include <raft/core/comms.hpp>

#include <thrust/functional.h>

namespace cugraph {
namespace reduce_op {

// Guidance on writing a custom reduction operator.
// 1. It is required to add an "using value_type = type_of_the_reduced_values" statement.
// 2. A custom reduction operator MUST be side-effect free. We use thrust::reduce internally to
// implement reductions in multiple primitives. The current (version 1.16)  implementation of thrust
// reduce rounds up the number of invocations based on the CUDA block size and discards the values
// outside the valid range.
// 3. If the return value of the reduction operator is solely determined by input argument values,
// define the pure function static member variable (i.e. "static constexpr pure_function = true;").
// This may enable better performance in multi-GPU as this flag indicates that the reduction
// operator can be executed in any GPU (this sometimes enable hierarchical reduction reducing
// communication volume & peak memory usage).
// 4. For simple reduction operations with a matching raft::comms::op_t value, specify the
// compatible_raft_comms_op static member variable (e.g. "static constexpr raft::comms::op_t
// compatible_raft_comms_op = raft::comms::op_t::MIN"). This often enables direct use of highly
// optimized the NCCL reduce functions instead of relying on a less efficient gather based reduction
// mechanism (we may implement a basic tree-based reduction mechanism in the future to improve the
// efficiency but this is still expected to be slower than the NCCL reduction).
// 5. Defining the identity_element static member variable (e.g. "inline static T const
// identity_element = T{}") potentially improves performance as well by avoiding special treatments
// for tricky corner cases.
// 6. See the pre-defined reduction operators below as examples.

// in case there is no payload to reduce
struct null {
  using value_type = void;
};

// Binary reduction operator selecting any of the two input arguments, T should be arithmetic types
// or thrust tuple of arithmetic types.
template <typename T>
struct any {
  using value_type                    = T;
  static constexpr bool pure_function = true;  // this can be called in any process

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return lhs; }
};

// Binary reduction operator selecting the minimum element of the two input arguments (using
// operator <), T should be arithmetic types or thrust tuple of arithmetic types.
template <typename T>
struct minimum {
  using value_type                    = T;
  static constexpr bool pure_function = true;  // this can be called in any process
  static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MIN;
  inline static T const identity_element                      = max_identity_element<T>();

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
};

// Binary reduction operator selecting the maximum element of the two input arguments (using
// operator <), T should be arithmetic types or thrust tuple of arithmetic types.
template <typename T>
struct maximum {
  using value_type                    = T;
  static constexpr bool pure_function = true;  // this can be called in any process
  static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MAX;
  inline static T const identity_element                      = min_identity_element<T>();

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const
  {
    return lhs < rhs ? rhs : lhs;
  }
};

// Binary reduction operator summing the two input arguments, T should be arithmetic types or thrust
// tuple of arithmetic types.
template <typename T>
struct plus {
  using value_type                    = T;
  static constexpr bool pure_function = true;  // this can be called in any process
  static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::SUM;
  inline static T const identity_element                      = T{};
  property_op<T, thrust::plus> op{};

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return op(lhs, rhs); }
};

template <typename ReduceOp, typename = raft::comms::op_t>
struct has_compatible_raft_comms_op : std::false_type {
};

template <typename ReduceOp>
struct has_compatible_raft_comms_op<ReduceOp,
                                    std::remove_cv_t<decltype(ReduceOp::compatible_raft_comms_op)>>
  : std::true_type {
};

template <typename ReduceOp>
inline constexpr bool has_compatible_raft_comms_op_v =
  has_compatible_raft_comms_op<ReduceOp>::value;

template <typename ReduceOp, typename = typename ReduceOp::value_type>
struct has_identity_element : std::false_type {
};

template <typename ReduceOp>
struct has_identity_element<ReduceOp, std::remove_cv_t<decltype(ReduceOp::identity_element)>>
  : std::true_type {
};

template <typename ReduceOp>
inline constexpr bool has_identity_element_v = has_identity_element<ReduceOp>::value;

template <typename ReduceOp, typename Iterator>
__device__ std::enable_if_t<has_compatible_raft_comms_op_v<ReduceOp>, void> atomic_reduce(
  Iterator iter, typename thrust::iterator_traits<Iterator>::value_type value)
{
  static_assert(std::is_same_v<typename ReduceOp::value_type,
                               typename thrust::iterator_traits<Iterator>::value_type>);
  static_assert(
    (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::SUM) ||
    (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::MIN) ||
    (ReduceOp::compatible_raft_comms_op ==
     raft::comms::op_t::MAX));  // currently, only (element-wise) sum, min, and max are supported.

  if constexpr (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::SUM) {
    atomic_add_edge_op_result(iter, value);
  } else if constexpr (ReduceOp::compatible_raft_comms_op == raft::comms::op_t::MIN) {
    atomic_min_edge_op_result(iter, value);
  } else {
    atomic_max_edge_op_result(iter, value);
  }
}

}  // namespace reduce_op
}  // namespace cugraph

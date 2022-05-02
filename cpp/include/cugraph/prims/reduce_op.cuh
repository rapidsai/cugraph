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

#include <cugraph/prims/property_op_utils.cuh>

#include <raft/comms/comms.hpp>

namespace cugraph {
namespace reduce_op {

// in case there is no payload to reduce
struct null {
  using type = void;
};

// Binary reduction operator selecting any of the two input arguments, T should be arithmetic types
// or thrust tuple of arithmetic types.
template <typename T>
struct any {
  using type = T;
  // Note that every reduction function should be side-effect free (to work with thrust::reduce, the
  // current (version 1.16)  implementation of thrust reduction call rounds up the number of
  // invocations based on the block size and discards the values outside the valid range). Set
  // pure_function to true if the reduction operator return value is solely determined by input
  // argument values in addition.
  static constexpr bool pure_function = true;  // this can be called in any process

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return lhs; }
};

// Binary reduction operator selecting the minimum element of the two input arguments (using
// operator <), T should be arithmetic types or thrust tuple of arithmetic types.
template <typename T>
struct minimum {
  using type = T;
  // Note that every reduction function should be side-effect free (to work with thrust::reduce, the
  // current (version 1.16)  implementation of thrust reduction call rounds up the number of
  // invocations based on the block size and discards the values outside the valid range). Set
  // pure_function to true if the reduction operator return value is solely determined by input
  // argument values in addition.
  static constexpr bool pure_function = true;  // this can be called in any process
  static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MIN;
  inline static T const identity_element                      = min_identity_element<T>();

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return lhs < rhs ? lhs : rhs; }
};

// Binary reduction operator selecting the maximum element of the two input arguments (using
// operator <), T should be arithmetic types or thrust tuple of arithmetic types.
template <typename T>
struct maximum {
  using type = T;
  // Note that every reduction function should be side-effect free (to work with thrust::reduce, the
  // current (version 1.16)  implementation of thrust reduction call rounds up the number of
  // invocations based on the block size and discards the values outside the valid range). Set
  // pure_function to true if the reduction operator return value is solely determined by input
  // argument values in addition.
  static constexpr bool pure_function = true;  // this can be called in any process
  static constexpr raft::comms::op_t compatible_raft_comms_op = raft::comms::op_t::MAX;
  inline static T const identity_element                      = max_identity_element<T>();

  __host__ __device__ T operator()(T const& lhs, T const& rhs) const { return lhs < rhs ? rhs : lhs; }
};

// Binary reduction operator summing the two input arguments, T should be arithmetic types or thrust
// tuple of arithmetic types.
template <typename T>
struct plus {
  using type = T;
  // Note that every reduction function should be side-effect free (to work with thrust::reduce, the
  // current (version 1.16)  implementation of thrust reduction call rounds up the number of
  // invocations based on the block size and discards the values outside the valid range). Set
  // pure_function to true if the reduction operator return value is solely determined by input
  // argument values in addition.
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

template <typename ReduceOp, typename = typename ReduceOp::type>
struct has_identity_element : std::false_type {
};

template <typename ReduceOp>
struct has_identity_element<ReduceOp, std::remove_cv_t<decltype(ReduceOp::identity_element)>>
  : std::true_type {
};

template <typename ReduceOp>
inline constexpr bool has_identity_element_v = has_identity_element<ReduceOp>::value;

}  // namespace reduce_op
}  // namespace cugraph

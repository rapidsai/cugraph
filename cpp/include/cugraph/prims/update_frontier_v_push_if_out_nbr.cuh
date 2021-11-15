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

#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

int32_t constexpr update_frontier_v_push_if_out_nbr_for_all_block_size = 512;

// we cannot use std::iterator_traits<Iterator>::value_type if Iterator is void* (reference to void
// is not allowed)
template <typename PayloadIterator, typename Enable = void>
struct optional_payload_buffer_value_type_t;

template <typename PayloadIterator>
struct optional_payload_buffer_value_type_t<
  PayloadIterator,
  std::enable_if_t<!std::is_same_v<PayloadIterator, void*>>> {
  using value = typename std::iterator_traits<PayloadIterator>::value_type;
};

template <typename PayloadIterator>
struct optional_payload_buffer_value_type_t<
  PayloadIterator,
  std::enable_if_t<std::is_same_v<PayloadIterator, void*>>> {
  using value = void;
};

// FIXME: to silence the spurious warning (missing return statement ...) due to the nvcc bug
// (https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-
// function-in-constexpr-if-fun)
#if 1
template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
std::byte allocate_optional_payload_buffer(size_t size, cudaStream_t stream)
{
  return std::byte{0};  // dummy
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
auto allocate_optional_payload_buffer(size_t size, cudaStream_t stream)
{
  return allocate_dataframe_buffer<payload_t>(size, stream);
}

template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
void* get_optional_payload_buffer_begin(std::byte& optional_payload_buffer)
{
  return static_cast<void*>(nullptr);
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
auto get_optional_payload_buffer_begin(
  std::add_lvalue_reference_t<decltype(allocate_dataframe_buffer<payload_t>(
    size_t{0}, cudaStream_t{nullptr}))> optional_payload_buffer)
{
  return get_dataframe_buffer_begin(optional_payload_buffer);
}
#else
auto allocate_optional_payload_buffer = [](size_t size, cudaStream_t stream) {
  if constexpr (std::is_same_v<payload_t, void>) {
    return std::byte{0};  // dummy
  } else {
    return allocate_dataframe_buffer<payload_t>(size, stream);
  }
};

auto get_optional_payload_buffer_begin = [](auto& optional_payload_buffer) {
  if constexpr (std::is_same_v<payload_t, void>) {
    return static_cast<std::byte*>(nullptr);
  } else {
    return get_dataframe_buffer_begin(optional_payload_buffer);
  }
};
#endif

// FIXME: a temporary workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
// in the else part in if constexpr else statement that involves device lambda
template <typename vertex_t,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp,
          typename key_t,
          bool multi_gpu>
struct call_v_op_t {
  VertexValueInputIterator vertex_value_input_first{};
  VertexValueOutputIterator vertex_value_output_first{};
  VertexOp v_op{};
  vertex_partition_device_view_t<vertex_t, multi_gpu> vertex_partition{};
  size_t invalid_bucket_idx;

  template <typename key_type = key_t, typename vertex_type = vertex_t>
  __device__ std::enable_if_t<std::is_same_v<key_type, vertex_type>, uint8_t> operator()(
    key_t key) const
  {
    auto v_offset    = vertex_partition.get_local_vertex_offset_from_vertex_nocheck(key);
    auto v_val       = *(vertex_value_input_first + v_offset);
    auto v_op_result = v_op(key, v_val);
    if (v_op_result) {
      *(vertex_value_output_first + v_offset) = thrust::get<1>(*v_op_result);
      return static_cast<uint8_t>(thrust::get<0>(*v_op_result));
    } else {
      return std::numeric_limits<uint8_t>::max();
    }
  }

  template <typename key_type = key_t, typename vertex_type = vertex_t>
  __device__ std::enable_if_t<!std::is_same_v<key_type, vertex_type>, uint8_t> operator()(
    key_t key) const
  {
    auto v_offset =
      vertex_partition.get_local_vertex_offset_from_vertex_nocheck(thrust::get<0>(key));
    auto v_val       = *(vertex_value_input_first + v_offset);
    auto v_op_result = v_op(key, v_val);
    if (v_op_result) {
      *(vertex_value_output_first + v_offset) = thrust::get<1>(*v_op_result);
      return static_cast<uint8_t>(thrust::get<0>(*v_op_result));
    } else {
      return std::numeric_limits<uint8_t>::max();
    }
  }
};

// FIXME: a temporary workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
// after if constexpr else statement that involves device lambda (bug report submitted)
template <typename key_t>
struct check_invalid_bucket_idx_t {
  __device__ bool operator()(thrust::tuple<uint8_t, key_t> pair)
  {
    return thrust::get<0>(pair) == std::numeric_limits<uint8_t>::max();
  }
};

template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__device__ void push_if_buffer_element(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu>& matrix_partition,
  typename std::iterator_traits<BufferKeyOutputIterator>::value_type key,
  typename GraphViewType::vertex_type row_offset,
  typename GraphViewType::vertex_type col,
  typename GraphViewType::weight_type weight,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using key_t    = typename std::iterator_traits<BufferKeyOutputIterator>::value_type;
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  auto col_offset  = matrix_partition.get_minor_offset_from_minor_nocheck(col);
  auto e_op_result = evaluate_edge_op<GraphViewType,
                                      key_t,
                                      AdjMatrixRowValueInputWrapper,
                                      AdjMatrixColValueInputWrapper,
                                      EdgeOp>()
                       .compute(key,
                                col,
                                weight,
                                adj_matrix_row_value_input.get(row_offset),
                                adj_matrix_col_value_input.get(col_offset),
                                e_op);
  if (e_op_result) {
    static_assert(sizeof(unsigned long long int) == sizeof(size_t));
    auto buffer_idx = atomicAdd(reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                static_cast<unsigned long long int>(1));
    if constexpr (std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>) {
      *(buffer_key_output_first + buffer_idx) = col;
    } else if constexpr (std::is_same_v<key_t, vertex_t> && !std::is_same_v<payload_t, void>) {
      *(buffer_key_output_first + buffer_idx)     = col;
      *(buffer_payload_output_first + buffer_idx) = *e_op_result;
    } else if constexpr (!std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>) {
      *(buffer_key_output_first + buffer_idx) = thrust::make_tuple(col, *e_op_result);
    } else {
      *(buffer_key_output_first + buffer_idx) =
        thrust::make_tuple(col, thrust::get<0>(*e_op_result));
      *(buffer_payload_output_first + buffer_idx) = thrust::get<1>(*e_op_result);
    }
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_hypersparse(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_hypersparse_first,
  KeyIterator key_first,
  KeyIterator key_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto row_start_offset =
    static_cast<size_t>(major_hypersparse_first - matrix_partition.get_major_first());
  auto idx = static_cast<size_t>(tid);

  auto dcs_nzd_vertices     = *(matrix_partition.get_dcs_nzd_vertices());
  auto dcs_nzd_vertex_count = *(matrix_partition.get_dcs_nzd_vertex_count());

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key = *(key_first + idx);
    vertex_t row{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      row = key;
    } else {
      row = thrust::get<0>(key);
    }
    auto row_hypersparse_idx = matrix_partition.get_major_hypersparse_idx_from_major_nocheck(row);
    if (row_hypersparse_idx) {
      auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
      auto row_idx    = row_start_offset + *row_hypersparse_idx;
      vertex_t const* indices{nullptr};
      thrust::optional<weight_t const*> weights{thrust::nullopt};
      edge_t local_out_degree{};
      thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_idx);
      for (edge_t i = 0; i < local_out_degree; ++i) {
        push_if_buffer_element<GraphViewType>(matrix_partition,
                                              key,
                                              row_offset,
                                              indices[i],
                                              weights ? (*weights)[i] : weight_t{1.0},
                                              adj_matrix_row_value_input,
                                              adj_matrix_col_value_input,
                                              buffer_key_output_first,
                                              buffer_payload_output_first,
                                              buffer_idx_ptr,
                                              e_op);
      }
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_low_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key = *(key_first + idx);
    vertex_t row{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      row = key;
    } else {
      row = thrust::get<0>(key);
    }
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (edge_t i = 0; i < local_out_degree; ++i) {
      push_if_buffer_element<GraphViewType>(matrix_partition,
                                            key,
                                            row_offset,
                                            indices[i],
                                            weights ? (*weights)[i] : weight_t{1.0},
                                            adj_matrix_row_value_input,
                                            adj_matrix_col_value_input,
                                            buffer_key_output_first,
                                            buffer_payload_output_first,
                                            buffer_idx_ptr,
                                            e_op);
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_mid_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(update_frontier_v_push_if_out_nbr_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key = *(key_first + idx);
    vertex_t row{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      row = key;
    } else {
      row = thrust::get<0>(key);
    }
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (edge_t i = lane_id; i < local_out_degree; i += raft::warp_size()) {
      push_if_buffer_element<GraphViewType>(matrix_partition,
                                            key,
                                            row_offset,
                                            indices[i],
                                            weights ? (*weights)[i] : weight_t{1.0},
                                            adj_matrix_row_value_input,
                                            adj_matrix_col_value_input,
                                            buffer_key_output_first,
                                            buffer_payload_output_first,
                                            buffer_idx_ptr,
                                            e_op);
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_high_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto idx = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key = *(key_first + idx);
    vertex_t row{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      row = key;
    } else {
      row = thrust::get<0>(key);
    }
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (edge_t i = threadIdx.x; i < local_out_degree; i += blockDim.x) {
      push_if_buffer_element<GraphViewType>(matrix_partition,
                                            key,
                                            row_offset,
                                            indices[i],
                                            weights ? (*weights)[i] : weight_t{1.0},
                                            adj_matrix_row_value_input,
                                            adj_matrix_col_value_input,
                                            buffer_key_output_first,
                                            buffer_payload_output_first,
                                            buffer_idx_ptr,
                                            e_op);
    }

    idx += gridDim.x;
  }
}

template <typename BufferKeyOutputIterator, typename BufferPayloadOutputIterator, typename ReduceOp>
size_t sort_and_reduce_buffer_elements(raft::handle_t const& handle,
                                       BufferKeyOutputIterator buffer_key_output_first,
                                       BufferPayloadOutputIterator buffer_payload_output_first,
                                       size_t num_buffer_elements,
                                       ReduceOp reduce_op)
{
  using key_t = typename std::iterator_traits<BufferKeyOutputIterator>::value_type;
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  auto execution_policy = handle.get_thrust_policy();
  if constexpr (std::is_same_v<payload_t, void>) {
    thrust::sort(
      execution_policy, buffer_key_output_first, buffer_key_output_first + num_buffer_elements);
  } else {
    thrust::sort_by_key(execution_policy,
                        buffer_key_output_first,
                        buffer_key_output_first + num_buffer_elements,
                        buffer_payload_output_first);
  }

  size_t num_reduced_buffer_elements{};
  if constexpr (std::is_same_v<payload_t, void>) {
    auto it = thrust::unique(
      execution_policy, buffer_key_output_first, buffer_key_output_first + num_buffer_elements);
    num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(buffer_key_output_first, it));
  } else if constexpr (std::is_same<ReduceOp, reduce_op::any<typename ReduceOp::type>>::value) {
    // FIXME: if ReducOp is any, we may have a cheaper alternative than sort & uique (i.e. discard
    // non-first elements)
    auto it = thrust::unique_by_key(execution_policy,
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first);
    num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(buffer_key_output_first, thrust::get<0>(it)));
  } else {
    // FIXME: better avoid temporary buffer or at least limit the maximum buffer size (if we adopt
    // CUDA cooperative group https://devblogs.nvidia.com/cooperative-groups and global sync(), we
    // can use aggregate shared memory as a temporary buffer, or we can limit the buffer size, and
    // split one thrust::reduce_by_key call to multiple thrust::reduce_by_key calls if the
    // temporary buffer size exceeds the maximum buffer size (may be definied as percentage of the
    // system HBM size or a function of the maximum number of threads in the system))
    // FIXME: actually, we can find how many unique keys are here by now.
    // FIXME: if GraphViewType::is_multi_gpu is true, this should be executed on the GPU holding
    // the vertex unless reduce_op is a pure function.
    rmm::device_uvector<key_t> keys(num_buffer_elements, handle.get_stream());
    auto value_buffer =
      allocate_dataframe_buffer<payload_t>(num_buffer_elements, handle.get_stream());
    auto it = thrust::reduce_by_key(execution_policy,
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first,
                                    keys.begin(),
                                    get_dataframe_buffer_begin(value_buffer),
                                    thrust::equal_to<key_t>(),
                                    reduce_op);
    num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(keys.begin(), thrust::get<0>(it)));
    // FIXME: this copy can be replaced by move
    thrust::copy(execution_policy,
                 keys.begin(),
                 keys.begin() + num_reduced_buffer_elements,
                 buffer_key_output_first);
    thrust::copy(execution_policy,
                 get_dataframe_buffer_begin(value_buffer),
                 get_dataframe_buffer_begin(value_buffer) + num_reduced_buffer_elements,
                 buffer_payload_output_first);
  }

  return num_reduced_buffer_elements;
}

}  // namespace detail

template <typename GraphViewType, typename VertexFrontierType>
typename GraphViewType::edge_type compute_num_out_nbrs_from_frontier(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierType const& frontier,
  size_t cur_frontier_bucket_idx)
{
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename VertexFrontierType::key_type;

  edge_t ret{0};

  auto const& cur_frontier_bucket = frontier.get_bucket(cur_frontier_bucket_idx);
  vertex_t const* local_frontier_vertex_first{nullptr};
  vertex_t const* local_frontier_vertex_last{nullptr};
  if constexpr (std::is_same_v<key_t, vertex_t>) {
    local_frontier_vertex_first = cur_frontier_bucket.begin();
    local_frontier_vertex_last  = cur_frontier_bucket.end();
  } else {
    local_frontier_vertex_first = thrust::get<0>(cur_frontier_bucket.begin().get_iterator_tuple());
    local_frontier_vertex_last  = thrust::get<0>(cur_frontier_bucket.end().get_iterator_tuple());
  }

  std::vector<size_t> local_frontier_sizes{};
  if (GraphViewType::is_multi_gpu) {
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    local_frontier_sizes =
      host_scalar_allgather(col_comm, cur_frontier_bucket.size(), handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(cur_frontier_bucket.size())};
  }
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    auto execution_policy = handle.get_thrust_policy();
    if (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      rmm::device_uvector<vertex_t> frontier_vertices(local_frontier_sizes[i],
                                                      handle.get_stream_view());
      device_bcast(col_comm,
                   local_frontier_vertex_first,
                   frontier_vertices.data(),
                   local_frontier_sizes[i],
                   static_cast<int>(i),
                   handle.get_stream());

      auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
      auto use_dcs =
        segment_offsets
          ? ((*segment_offsets).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
          : false;

      ret +=
        use_dcs
          ? thrust::transform_reduce(
              execution_policy,
              frontier_vertices.begin(),
              frontier_vertices.end(),
              [matrix_partition,
               major_hypersparse_first =
                 matrix_partition.get_major_first() +
                 (*segment_offsets)
                   [detail::num_sparse_segments_per_vertex_partition]] __device__(auto major) {
                if (major < major_hypersparse_first) {
                  auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
                  return matrix_partition.get_local_degree(major_offset);
                } else {
                  auto major_hypersparse_idx =
                    matrix_partition.get_major_hypersparse_idx_from_major_nocheck(major);
                  return major_hypersparse_idx
                           ? matrix_partition.get_local_degree(
                               matrix_partition.get_major_offset_from_major_nocheck(
                                 major_hypersparse_first) +
                               *major_hypersparse_idx)
                           : edge_t{0};
                }
              },
              edge_t{0},
              thrust::plus<edge_t>())
          : thrust::transform_reduce(
              execution_policy,
              frontier_vertices.begin(),
              frontier_vertices.end(),
              [matrix_partition] __device__(auto major) {
                auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
                return matrix_partition.get_local_degree(major_offset);
              },
              edge_t{0},
              thrust::plus<edge_t>());
    } else {
      assert(i == 0);
      ret += thrust::transform_reduce(
        execution_policy,
        local_frontier_vertex_first,
        local_frontier_vertex_last,
        [matrix_partition] __device__(auto major) {
          auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
          return matrix_partition.get_local_degree(major_offset);
        },
        edge_t{0},
        thrust::plus<edge_t>());
    }
  }

  return ret;
}

// FIXME: this documentation needs to be updated due to (tagged-)vertex support
/**
 * @brief Update (tagged-)vertex frontier and (tagged-)vertex property values iterating over the
 * outgoing edges from the frontier.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierType Type of the vertex frontier class which abstracts vertex frontier
 * managements.
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex property variables.
 * @tparam VertexOp Type of the binary vertex operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontier class object for vertex frontier managements. This object includes
 * multiple bucket objects.
 * @param cur_frontier_bucket_idx Index of the VertexFrontier bucket holding vertices for the
 * current iteration.
 * @param next_frontier_bucket_indices Indices of the VertexFrontier buckets to store new frontier
 * vertices for the next iteration.
 * @param adj_matrix_row_value_input Device-copyable wrapper used to access row input properties
 * (for the rows assigned to this process in multi-GPU). Use either
 * cugraph::row_properties_t::device_view() (if @p e_op needs to access row properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access row properties). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param adj_matrix_col_value_input Device-copyable wrapper used to access column input properties
 * (for the columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view() (if @p e_op needs to access column properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access column properties). Use
 * copy_to_adj_matrix_col to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns a value to be reduced the @p reduce_op.
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 * @param v_op Ternary operator takes (tagged-)vertex ID, *(@p vertex_value_input_first + i) (where
 * i is [0, @p graph_view.get_number_of_local_vertices())) and reduced value of the @p e_op outputs
 * for this vertex and returns the target bucket index (for frontier update) and new verrtex
 * property values (to update *(@p vertex_value_output_first + i)). The target bucket index should
 * either be VertexFrontierType::kInvalidBucketIdx or an index in @p next_frontier_bucket_indices.
 */
template <typename GraphViewType,
          typename VertexFrontierType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp>
void update_frontier_v_push_if_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierType& frontier,
  size_t cur_frontier_bucket_idx,
  std::vector<size_t> const& next_frontier_bucket_indices,
  // FIXME: if vertices in the frontier are tagged, we should have an option to access with (vertex,
  // tag) pair (currently we can access only with vertex, we may use cuco::static_map for this
  // purpose)
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  EdgeOp e_op,
  ReduceOp reduce_op,
  // FIXME: if vertices in the frontier are tagged, we should have an option to access with (vertex,
  // tag) pair (currently we can access only with vertex, we may use cuco::static_map for this
  // purpose)
  VertexValueInputIterator vertex_value_input_first,
  // FIXME: if vertices in the frontier are tagged, we should have an option to access with (vertex,
  // tag) pair (currently we can access only with vertex, we may use cuco::static_map for this
  // purpose)
  // FIXME: currently, it is undefined behavior if vertices in the frontier are tagged and the same
  // vertex property is updated by multiple v_op invocations with the same vertex but with different
  // tags.
  VertexValueOutputIterator vertex_value_output_first,
  // FIXME: this takes (tagged-)vertex ID in addition, think about consistency with the other
  // primitives.
  VertexOp v_op)
{
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  using weight_t  = typename GraphViewType::weight_type;
  using key_t     = typename VertexFrontierType::key_type;
  using payload_t = typename ReduceOp::type;

  auto frontier_key_first = frontier.get_bucket(cur_frontier_bucket_idx).begin();
  auto frontier_key_last  = frontier.get_bucket(cur_frontier_bucket_idx).end();

  // 1. fill the buffer

  auto key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
  auto payload_buffer =
    detail::allocate_optional_payload_buffer<payload_t>(size_t{0}, handle.get_stream());
  rmm::device_scalar<size_t> buffer_idx(size_t{0}, handle.get_stream());
  std::vector<size_t> local_frontier_sizes{};
  if (GraphViewType::is_multi_gpu) {
    auto& col_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    local_frontier_sizes = host_scalar_allgather(
      col_comm,
      static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
      handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(
      static_cast<vertex_t>(thrust::distance(frontier_key_first, frontier_key_last)))};
  }
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    auto matrix_partition_frontier_key_buffer =
      allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
    vertex_t matrix_partition_frontier_size = static_cast<vertex_t>(local_frontier_sizes[i]);
    if (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      resize_dataframe_buffer(
        matrix_partition_frontier_key_buffer, matrix_partition_frontier_size, handle.get_stream());

      device_bcast(col_comm,
                   frontier_key_first,
                   get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer),
                   matrix_partition_frontier_size,
                   i,
                   handle.get_stream());
    } else {
      resize_dataframe_buffer(
        matrix_partition_frontier_key_buffer, matrix_partition_frontier_size, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   frontier_key_first,
                   frontier_key_last,
                   get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer));
    }

    vertex_t const* matrix_partition_frontier_row_first{nullptr};
    vertex_t const* matrix_partition_frontier_row_last{nullptr};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      matrix_partition_frontier_row_first =
        get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer);
      matrix_partition_frontier_row_last =
        get_dataframe_buffer_end(matrix_partition_frontier_key_buffer);
    } else {
      matrix_partition_frontier_row_first = thrust::get<0>(
        get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer).get_iterator_tuple());
      matrix_partition_frontier_row_last = thrust::get<0>(
        get_dataframe_buffer_end(matrix_partition_frontier_key_buffer).get_iterator_tuple());
    }

    auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
    auto use_dcs =
      segment_offsets
        ? ((*segment_offsets).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
        : false;

    auto execution_policy = handle.get_thrust_policy();
    auto max_pushes =
      use_dcs ? thrust::transform_reduce(
                  execution_policy,
                  matrix_partition_frontier_row_first,
                  matrix_partition_frontier_row_last,
                  [matrix_partition,
                   major_hypersparse_first =
                     matrix_partition.get_major_first() +
                     (*segment_offsets)
                       [detail::num_sparse_segments_per_vertex_partition]] __device__(auto row) {
                    if (row < major_hypersparse_first) {
                      auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
                      return matrix_partition.get_local_degree(row_offset);
                    } else {
                      auto row_hypersparse_idx =
                        matrix_partition.get_major_hypersparse_idx_from_major_nocheck(row);
                      return row_hypersparse_idx
                               ? matrix_partition.get_local_degree(
                                   matrix_partition.get_major_offset_from_major_nocheck(
                                     major_hypersparse_first) +
                                   *row_hypersparse_idx)
                               : edge_t{0};
                    }
                  },
                  edge_t{0},
                  thrust::plus<edge_t>())
              : thrust::transform_reduce(
                  execution_policy,
                  matrix_partition_frontier_row_first,
                  matrix_partition_frontier_row_last,
                  [matrix_partition] __device__(auto row) {
                    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
                    return matrix_partition.get_local_degree(row_offset);
                  },
                  edge_t{0},
                  thrust::plus<edge_t>());

    // FIXME: This is highly pessimistic for single GPU (and multi-GPU as well if we maintain
    // additional per column data for filtering in e_op). If we can pause & resume execution if
    // buffer needs to be increased (and if we reserve address space to avoid expensive
    // reallocation;
    // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/), we can
    // start with a smaller buffer size (especially when the frontier size is large).
    // for special cases when we can assure that there is no more than one push per destination
    // (e.g. if cugraph::reduce_op::any is used), we can limit the buffer size to
    // std::min(max_pushes, matrix_partition.get_minor_size()).
    // For Volta+, we can limit the buffer size to std::min(max_pushes,
    // matrix_partition.get_minor_size()) if the reduction operation is a pure function if we use
    // locking.
    // FIXME: if i != 0, this will require costly reallocation if we don't use the new CUDA feature
    // to reserve address space.
    auto new_buffer_size = buffer_idx.value(handle.get_stream()) + max_pushes;
    resize_dataframe_buffer(key_buffer, new_buffer_size, handle.get_stream());
    if constexpr (!std::is_same_v<payload_t, void>) {
      resize_dataframe_buffer(payload_buffer, new_buffer_size, handle.get_stream());
    }

    auto matrix_partition_row_value_input = adj_matrix_row_value_input;
    auto matrix_partition_col_value_input = adj_matrix_col_value_input;
    matrix_partition_row_value_input.set_local_adj_matrix_partition_idx(i);

    if (segment_offsets) {
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
      std::vector<vertex_t> h_thresholds(detail::num_sparse_segments_per_vertex_partition +
                                         (use_dcs ? 1 : 0) - 1);
      h_thresholds[0] = matrix_partition.get_major_first() + (*segment_offsets)[1];
      h_thresholds[1] = matrix_partition.get_major_first() + (*segment_offsets)[2];
      if (use_dcs) { h_thresholds[2] = matrix_partition.get_major_first() + (*segment_offsets)[3]; }
      rmm::device_uvector<vertex_t> d_thresholds(h_thresholds.size(), handle.get_stream());
      raft::update_device(
        d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> d_offsets(d_thresholds.size(), handle.get_stream());
      thrust::lower_bound(handle.get_thrust_policy(),
                          matrix_partition_frontier_row_first,
                          matrix_partition_frontier_row_last,
                          d_thresholds.begin(),
                          d_thresholds.end(),
                          d_offsets.begin());
      std::vector<vertex_t> h_offsets(d_offsets.size());
      raft::update_host(h_offsets.data(), d_offsets.data(), d_offsets.size(), handle.get_stream());
      CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
      h_offsets.push_back(matrix_partition_frontier_size);
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      if (h_offsets[0] > 0) {
        raft::grid_1d_block_t update_grid(
          h_offsets[0],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_row_for_all_nbr_high_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer),
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[0],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
      if (h_offsets[1] - h_offsets[0] > 0) {
        raft::grid_1d_warp_t update_grid(
          h_offsets[1] - h_offsets[0],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_row_for_all_nbr_mid_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[0],
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[1],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
      if (h_offsets[2] - h_offsets[1] > 0) {
        raft::grid_1d_thread_t update_grid(
          h_offsets[2] - h_offsets[1],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_row_for_all_nbr_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[1],
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[2],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
      if (matrix_partition.get_dcs_nzd_vertex_count() && (h_offsets[3] - h_offsets[2] > 0)) {
        raft::grid_1d_thread_t update_grid(
          h_offsets[3] - h_offsets[2],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_row_for_all_nbr_hypersparse<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first() + (*segment_offsets)[3],
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[2],
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer) + h_offsets[3],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
    } else {
      if (matrix_partition_frontier_size > 0) {
        raft::grid_1d_thread_t update_grid(
          matrix_partition_frontier_size,
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_frontier_row_for_all_nbr_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            get_dataframe_buffer_begin(matrix_partition_frontier_key_buffer),
            get_dataframe_buffer_end(matrix_partition_frontier_key_buffer),
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
    }
  }

  // 2. reduce the buffer

  auto num_buffer_elements = detail::sort_and_reduce_buffer_elements(
    handle,
    get_dataframe_buffer_begin(key_buffer),
    detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
    buffer_idx.value(handle.get_stream()),
    reduce_op);
  if (GraphViewType::is_multi_gpu) {
    // FIXME: this step is unnecessary if row_comm_size== 1
    auto& comm               = handle.get_comms();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();

    std::vector<vertex_t> h_vertex_lasts(row_comm_size);
    for (size_t i = 0; i < h_vertex_lasts.size(); ++i) {
      h_vertex_lasts[i] = graph_view.get_vertex_partition_last(col_comm_rank * row_comm_size + i);
    }

    rmm::device_uvector<vertex_t> d_vertex_lasts(h_vertex_lasts.size(), handle.get_stream());
    raft::update_device(
      d_vertex_lasts.data(), h_vertex_lasts.data(), h_vertex_lasts.size(), handle.get_stream());
    rmm::device_uvector<edge_t> d_tx_buffer_last_boundaries(d_vertex_lasts.size(),
                                                            handle.get_stream());
    vertex_t const* row_first{nullptr};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      row_first = get_dataframe_buffer_begin(key_buffer);
    } else {
      row_first = thrust::get<0>(get_dataframe_buffer_begin(key_buffer).get_iterator_tuple());
    }
    thrust::lower_bound(handle.get_thrust_policy(),
                        row_first,
                        row_first + num_buffer_elements,
                        d_vertex_lasts.begin(),
                        d_vertex_lasts.end(),
                        d_tx_buffer_last_boundaries.begin());
    std::vector<edge_t> h_tx_buffer_last_boundaries(d_tx_buffer_last_boundaries.size());
    raft::update_host(h_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.size(),
                      handle.get_stream());
    handle.get_stream_view().synchronize();
    std::vector<size_t> tx_counts(h_tx_buffer_last_boundaries.size());
    std::adjacent_difference(
      h_tx_buffer_last_boundaries.begin(), h_tx_buffer_last_boundaries.end(), tx_counts.begin());

    auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
    std::tie(rx_key_buffer, std::ignore) = shuffle_values(
      row_comm, get_dataframe_buffer_begin(key_buffer), tx_counts, handle.get_stream());
    key_buffer = std::move(rx_key_buffer);

    if constexpr (!std::is_same_v<payload_t, void>) {
      auto rx_payload_buffer = allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
      std::tie(rx_payload_buffer, std::ignore) = shuffle_values(
        row_comm, get_dataframe_buffer_begin(payload_buffer), tx_counts, handle.get_stream());
      payload_buffer = std::move(rx_payload_buffer);
    }

    num_buffer_elements = detail::sort_and_reduce_buffer_elements(
      handle,
      get_dataframe_buffer_begin(key_buffer),
      detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
      size_dataframe_buffer(key_buffer),
      reduce_op);
  }

  // 3. update vertex properties and frontier

  if (num_buffer_elements > 0) {
    static_assert(VertexFrontierType::kNumBuckets <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(num_buffer_elements, handle.get_stream());

    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      graph_view.get_vertex_partition_view());

    if constexpr (!std::is_same_v<payload_t, void>) {
      auto key_payload_pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(get_dataframe_buffer_begin(key_buffer),
                           detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer)));
      thrust::transform(
        handle.get_thrust_policy(),
        key_payload_pair_first,
        key_payload_pair_first + num_buffer_elements,
        bucket_indices.begin(),
        [vertex_value_input_first,
         vertex_value_output_first,
         v_op,
         vertex_partition,
         invalid_bucket_idx = VertexFrontierType::kInvalidBucketIdx] __device__(auto pair) {
          auto key     = thrust::get<0>(pair);
          auto payload = thrust::get<1>(pair);
          vertex_t v_offset{};
          if constexpr (std::is_same_v<key_t, vertex_t>) {
            v_offset = vertex_partition.get_local_vertex_offset_from_vertex_nocheck(key);
          } else {
            v_offset =
              vertex_partition.get_local_vertex_offset_from_vertex_nocheck(thrust::get<0>(key));
          }
          auto v_val       = *(vertex_value_input_first + v_offset);
          auto v_op_result = v_op(key, v_val, payload);
          if (v_op_result) {
            *(vertex_value_output_first + v_offset) = thrust::get<1>(*v_op_result);
            return static_cast<uint8_t>(thrust::get<0>(*v_op_result));
          } else {
            return std::numeric_limits<uint8_t>::max();
          }
        });

      resize_dataframe_buffer(payload_buffer, size_t{0}, handle.get_stream());
      shrink_to_fit_dataframe_buffer(payload_buffer, handle.get_stream());
    } else {
      thrust::transform(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(key_buffer),
        get_dataframe_buffer_begin(key_buffer) + num_buffer_elements,
        bucket_indices.begin(),
        detail::call_v_op_t<vertex_t,
                            VertexValueInputIterator,
                            VertexValueOutputIterator,
                            VertexOp,
                            key_t,
                            GraphViewType::is_multi_gpu>{vertex_value_input_first,
                                                         vertex_value_output_first,
                                                         v_op,
                                                         vertex_partition,
                                                         VertexFrontierType::kInvalidBucketIdx});
    }

    auto bucket_key_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(bucket_indices.begin(), get_dataframe_buffer_begin(key_buffer)));
    bucket_indices.resize(
      thrust::distance(bucket_key_pair_first,
                       thrust::remove_if(handle.get_thrust_policy(),
                                         bucket_key_pair_first,
                                         bucket_key_pair_first + num_buffer_elements,
                                         detail::check_invalid_bucket_idx_t<key_t>())),
      handle.get_stream());
    resize_dataframe_buffer(key_buffer, bucket_indices.size(), handle.get_stream());
    bucket_indices.shrink_to_fit(handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());

    frontier.insert_to_buckets(bucket_indices.begin(),
                               bucket_indices.end(),
                               get_dataframe_buffer_begin(key_buffer),
                               next_frontier_bucket_indices);
  }
}

}  // namespace cugraph

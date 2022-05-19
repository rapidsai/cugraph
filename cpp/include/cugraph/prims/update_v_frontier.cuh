/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_tuple_utils.cuh>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

// FIXME: a temporary workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
// in the else part in if constexpr else statement that involves device lambda
template <typename vertex_t,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp,
          typename key_t,
          bool multi_gpu>
struct update_v_frontier_call_v_op_t {
  VertexValueInputIterator vertex_value_input_first{};
  VertexValueOutputIterator vertex_value_output_first{};
  VertexOp v_op{};
  vertex_t local_vertex_partition_range_first{};

  template <typename key_type = key_t, typename vertex_type = vertex_t>
  __device__ std::enable_if_t<std::is_same_v<key_type, vertex_type>, uint8_t> operator()(
    key_t key) const
  {
    auto v_offset    = key - local_vertex_partition_range_first;
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
    auto v_offset    = thrust::get<0>(key) - local_vertex_partition_range_first;
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

}  // namespace detail

template <typename GraphViewType,
          typename KeyBuffer,
          typename PayloadBuffer,
          typename VertexFrontierType,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp>
void update_v_frontier(raft::handle_t const& handle,
                       GraphViewType const& graph_view,
                       KeyBuffer&& key_buffer,
                       PayloadBuffer&& payload_buffer,
                       VertexFrontierType& frontier,
                       std::vector<size_t> const& next_frontier_bucket_indices,
                       VertexValueInputIterator vertex_value_input_first,
                       // FIXME: currently, it is undefined behavior if vertices in the frontier are
                       // tagged and the same vertex property is updated by multiple v_op
                       // invocations with the same vertex but with different tags.
                       VertexValueOutputIterator vertex_value_output_first,
                       VertexOp v_op,
                       bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using key_t =
    typename thrust::iterator_traits<decltype(get_dataframe_buffer_begin(key_buffer))>::value_type;
  using payload_t = typename thrust::iterator_traits<decltype(
    get_dataframe_buffer_begin(payload_buffer))>::value_type;

  static_assert(std::is_rvalue_reference_v<decltype(key_buffer)>);
  static_assert(std::is_rvalue_reference_v<decltype(payload_buffer)>);

  if (do_expensive_check) {}

  if (size_dataframe_buffer(key_buffer) > 0) {
    assert(frontier.num_buckets() <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(size_dataframe_buffer(key_buffer),
                                                handle.get_stream());

    auto key_payload_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      get_dataframe_buffer_begin(key_buffer), get_dataframe_buffer_begin(payload_buffer)));
    thrust::transform(handle.get_thrust_policy(),
                      key_payload_pair_first,
                      key_payload_pair_first + size_dataframe_buffer(key_buffer),
                      bucket_indices.begin(),
                      [vertex_value_input_first,
                       vertex_value_output_first,
                       v_op,
                       local_vertex_partition_range_first =
                         graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
                        auto key     = thrust::get<0>(pair);
                        auto payload = thrust::get<1>(pair);
                        vertex_t v_offset{};
                        if constexpr (std::is_same_v<key_t, vertex_t>) {
                          v_offset = key - local_vertex_partition_range_first;
                        } else {
                          v_offset = thrust::get<0>(key) - local_vertex_partition_range_first;
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

    auto bucket_key_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(bucket_indices.begin(), get_dataframe_buffer_begin(key_buffer)));
    bucket_indices.resize(
      thrust::distance(bucket_key_pair_first,
                       thrust::remove_if(handle.get_thrust_policy(),
                                         bucket_key_pair_first,
                                         bucket_key_pair_first + bucket_indices.size(),
                                         detail::check_invalid_bucket_idx_t<key_t>())),
      handle.get_stream());
    resize_dataframe_buffer(key_buffer, bucket_indices.size(), handle.get_stream());
    bucket_indices.shrink_to_fit(handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());

    frontier.insert_to_buckets(bucket_indices.begin(),
                               bucket_indices.end(),
                               get_dataframe_buffer_begin(key_buffer),
                               next_frontier_bucket_indices);

    resize_dataframe_buffer(key_buffer, 0, handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
  }
}

template <typename GraphViewType,
          typename KeyBuffer,
          typename VertexFrontierType,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp>
void update_v_frontier(raft::handle_t const& handle,
                       GraphViewType const& graph_view,
                       KeyBuffer&& key_buffer,
                       VertexFrontierType& frontier,
                       std::vector<size_t> const& next_frontier_bucket_indices,
                       VertexValueInputIterator vertex_value_input_first,
                       // FIXME: currently, it is undefined behavior if vertices in the frontier are
                       // tagged and the same vertex property is updated by multiple v_op
                       // invocations with the same vertex but with different tags.
                       VertexValueOutputIterator vertex_value_output_first,
                       VertexOp v_op,
                       bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using key_t =
    typename thrust::iterator_traits<decltype(get_dataframe_buffer_begin(key_buffer))>::value_type;

  static_assert(std::is_rvalue_reference_v<decltype(key_buffer)>);

  if (do_expensive_check) {}

  if (size_dataframe_buffer(key_buffer) > 0) {
    assert(frontier.num_buckets() <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(size_dataframe_buffer(key_buffer),
                                                handle.get_stream());

    thrust::transform(handle.get_thrust_policy(),
                      get_dataframe_buffer_begin(key_buffer),
                      get_dataframe_buffer_begin(key_buffer) + size_dataframe_buffer(key_buffer),
                      bucket_indices.begin(),
                      detail::update_v_frontier_call_v_op_t<vertex_t,
                                                            VertexValueInputIterator,
                                                            VertexValueOutputIterator,
                                                            VertexOp,
                                                            key_t,
                                                            GraphViewType::is_multi_gpu>{
                        vertex_value_input_first,
                        vertex_value_output_first,
                        v_op,
                        graph_view.local_vertex_partition_range_first()});

    auto bucket_key_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(bucket_indices.begin(), get_dataframe_buffer_begin(key_buffer)));
    bucket_indices.resize(
      thrust::distance(bucket_key_pair_first,
                       thrust::remove_if(handle.get_thrust_policy(),
                                         bucket_key_pair_first,
                                         bucket_key_pair_first + bucket_indices.size(),
                                         detail::check_invalid_bucket_idx_t<key_t>())),
      handle.get_stream());
    resize_dataframe_buffer(key_buffer, bucket_indices.size(), handle.get_stream());
    bucket_indices.shrink_to_fit(handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());

    frontier.insert_to_buckets(bucket_indices.begin(),
                               bucket_indices.end(),
                               get_dataframe_buffer_begin(key_buffer),
                               next_frontier_bucket_indices);

    resize_dataframe_buffer(key_buffer, 0, handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
  }
}

}  // namespace cugraph

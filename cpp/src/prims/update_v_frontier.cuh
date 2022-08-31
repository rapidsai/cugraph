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
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
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

template <typename vertex_t,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp,
          typename key_t,
          typename payload_t>
struct update_v_frontier_call_v_op_t {
  VertexValueInputIterator vertex_value_input_first{};
  VertexValueOutputIterator vertex_value_output_first{};
  VertexOp v_op{};
  vertex_t local_vertex_partition_range_first{};

  __device__ uint8_t operator()(thrust::tuple<key_t, payload_t> pair) const
  {
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
    if (thrust::get<1>(v_op_result)) {
      *(vertex_value_output_first + v_offset) = *(thrust::get<1>(v_op_result));
    }
    if (thrust::get<0>(v_op_result)) {
      assert(*(thrust::get<0>(v_op_result)) < std::numeric_limits<uint8_t>::max());
      return static_cast<uint8_t>(*(thrust::get<0>(v_op_result)));
    } else {
      return std::numeric_limits<uint8_t>::max();
    }
  }
};

template <typename vertex_t,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp,
          typename key_t>
struct update_v_frontier_call_v_op_t<vertex_t,
                                     VertexValueInputIterator,
                                     VertexValueOutputIterator,
                                     VertexOp,
                                     key_t,
                                     void> {
  VertexValueInputIterator vertex_value_input_first{};
  VertexValueOutputIterator vertex_value_output_first{};
  VertexOp v_op{};
  vertex_t local_vertex_partition_range_first{};

  __device__ uint8_t operator()(key_t key) const
  {
    vertex_t v_offset{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      v_offset = key - local_vertex_partition_range_first;
    } else {
      v_offset = thrust::get<0>(key) - local_vertex_partition_range_first;
    }
    auto v_val       = *(vertex_value_input_first + v_offset);
    auto v_op_result = v_op(key, v_val);
    if (thrust::get<1>(v_op_result)) {
      *(vertex_value_output_first + v_offset) = *(thrust::get<1>(v_op_result));
    }
    if (thrust::get<0>(v_op_result)) {
      assert(*(thrust::get<0>(v_op_result)) < std::numeric_limits<uint8_t>::max());
      return static_cast<uint8_t>(*(thrust::get<0>(v_op_result)));
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

/**
 * @brief Insert (tagged-)vertices to the vertex frontier and update vertex property values of the
 * newly inserted vertices .
 *
 * This primitive often works in pair with transform_reduce_v_frontier_outgoing_e_by_dst. This
 * version of update_v_frontier takes @p payload_buffer and @v_op takes a payload value in addition
 * to a (tagged-)vertex and a vertex property value as input arguments.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBuffer Type of the buffer storing (tagged-)vertices.
 * @tparam PayloadBuffer Type of the buffer storing payload values.
 * @tparam VertexFrontierType Type of the vertex frontier class which abstracts vertex frontier
 * managements.
 * @tparam VertexValueInputIterator Type of the iterator for input vertex property values.
 * @tparam VertexValueOutputIterator Type of the iterator for output vertex property variables.
 * @tparam VertexOp Type of the ternary vertex operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_buffer buffer object storing (tagged-)vertices to insert.
 * @param payload_buffer buffer object storing payload values for each (tagged-)vertices in the @p
 * key_buffer.
 * @param frontier VertexFrontierType class object for vertex frontier managements. This object
 * includes multiple bucket objects.
 * @param next_frontier_bucket_indices Indices of the vertex frontier buckets to store new frontier
 * (tagged-)vertices.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param v_op Ternary operator that takes (tagged-)vertex ID, *(@p vertex_value_input_first + i)
 * (where i is [0, @p graph_view.local_vertex_partition_range_size())) and the payload value for the
 * (tagged-)vertex ID and returns a tuple of 1) a thrust::optional object optionally storing a
 * bucket index and 2) a thrust::optional object optionally storing a new vertex property value. If
 * the first element of the returned tuple is thrust::nullopt, this (tagged-)vertex won't be
 * inserted to the vertex frontier. If the second element is thrust::nullopt, the vertex property
 * value for this vertex won't be updated. Note that it is currently undefined behavior if there are
 * multiple tagged-vertices with the same vertex ID (but with different tags) AND @p v_op results on
 * the tagged-vertices with the same vertex ID have more than one valid new vertex property values.
 */
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
                       // FIXME: currently, it is undefined behavior if there are more than one @p
                       // key_buffer elements with the same vertex ID and the same vertex property
                       // value is updated by multiple @p v_op invocations with the same vertex ID.
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

  std::for_each(next_frontier_bucket_indices.begin(),
                next_frontier_bucket_indices.end(),
                [&frontier](auto idx) {
                  CUGRAPH_EXPECTS(idx < frontier.num_buckets(),
                                  "Invalid input argument: invalid next bucket indices.");
                });

  if (do_expensive_check) {
    // currently, nothing to do
  }

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
                      detail::update_v_frontier_call_v_op_t<vertex_t,
                                                            VertexValueInputIterator,
                                                            VertexValueOutputIterator,
                                                            VertexOp,
                                                            key_t,
                                                            payload_t>{
                        vertex_value_input_first,
                        vertex_value_output_first,
                        v_op,
                        graph_view.local_vertex_partition_range_first()});

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

/**
 * @brief Insert (tagged-)vertices to the vertex frontier and update vertex property values of the
 * newly inserted vertices .
 *
 * This primitive often works in pair with transform_reduce_v_frontier_outgoing_e_by_dst. This
 * version of update_v_frontier does not take @p payload_buffer and @v_op takes a (tagged-)vertex
 * and a vertex property value as input arguments (no payload value in the input parameter list).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBuffer Type of the buffer storing (tagged-)vertices.
 * @tparam VertexFrontierType Type of the vertex frontier class which abstracts vertex frontier
 * managements.
 * @tparam VertexValueInputIterator Type of the iterator for input vertex property values.
 * @tparam VertexValueOutputIterator Type of the iterator for output vertex property variables.
 * @tparam VertexOp Type of the binary vertex operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_buffer buffer object storing (tagged-)vertices to insert.
 * @param frontier VertexFrontierType class object for vertex frontier managements. This object
 * includes multiple bucket objects.
 * @param next_frontier_bucket_indices Indices of the vertex frontier buckets to store new frontier
 * (tagged-)vertices.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param v_op Binary operator that takes (tagged-)vertex ID, and *(@p vertex_value_input_first + i)
 * (where i is [0, @p graph_view.local_vertex_partition_range_size())) and returns a tuple of 1) a
 * thrust::optional object optionally storing a bucket index and 2) a thrust::optional object
 * optionally storing a new vertex property value. If the first element of the returned tuple is
 * thrust::nullopt, this (tagged-)vertex won't be inserted to the vertex frontier. If the second
 * element is thrust::nullopt, the vertex property value for this vertex won't be updated. Note that
 * it is currently undefined behavior if there are multiple tagged-vertices with the same vertex ID
 * (but with different tags) AND @p v_op results on the tagged-vertices with the same vertex ID have
 * more than one valid new vertex property values.
 */
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
                       // FIXME: currently, it is undefined behavior if there are more than one @p
                       // key_buffer elements with the same vertex ID and the same vertex property
                       // value is updated by multiple @p v_op invocations with the same vertex ID.
                       VertexValueOutputIterator vertex_value_output_first,
                       VertexOp v_op,
                       bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using key_t =
    typename thrust::iterator_traits<decltype(get_dataframe_buffer_begin(key_buffer))>::value_type;

  static_assert(std::is_rvalue_reference_v<decltype(key_buffer)>);

  std::for_each(next_frontier_bucket_indices.begin(),
                next_frontier_bucket_indices.end(),
                [&frontier](auto idx) {
                  CUGRAPH_EXPECTS(idx < frontier.num_buckets(),
                                  "Invalid input argument: invalid next bucket indices.");
                });

  if (do_expensive_check) {
    // currently, nothing to do
  }

  if (size_dataframe_buffer(key_buffer) > 0) {
    assert(frontier.num_buckets() <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(size_dataframe_buffer(key_buffer),
                                                handle.get_stream());

    thrust::transform(
      handle.get_thrust_policy(),
      get_dataframe_buffer_begin(key_buffer),
      get_dataframe_buffer_begin(key_buffer) + size_dataframe_buffer(key_buffer),
      bucket_indices.begin(),
      detail::update_v_frontier_call_v_op_t<vertex_t,
                                            VertexValueInputIterator,
                                            VertexValueOutputIterator,
                                            VertexOp,
                                            key_t,
                                            void>{vertex_value_input_first,
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

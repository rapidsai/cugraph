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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

namespace cugraph {
namespace detail {

/**
 * @brief    Fill a buffer with uniformly distributed random values
 *
 * Fills a buffer with uniformly distributed random values between
 * the specified minimum and maximum values.
 *
 * @tparam      value_t      type of the value to operate on
 *
 * @param[in]   stream_view  stream view
 * @param[out]  d_value      device array to fill
 * @param[in]   size         number of elements in array
 * @param[in]   min_value    minimum value (inclusive)
 * @param[in]   max_value    maximum value (exclusive)
 * @param[in]   rng_state    The RngState instance holding pseudo-random number generator state.
 *
 */
template <typename value_t>
void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                         value_t* d_value,
                         size_t size,
                         value_t min_value,
                         value_t max_value,
                         raft::random::RngState& rng_state);

/**
 * @brief    Fill a buffer with a constant value
 *
 * @tparam      value_t      type of the value to operate on
 *
 * @param [in]  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[out]  d_value      device array to fill
 * @param[in]   size         number of elements in array
 * @param[in]   value        value
 *
 */
template <typename value_t>
void scalar_fill(raft::handle_t const& handle, value_t* d_value, size_t size, value_t value);

/**
 * @brief    Fill a buffer with a sequence of values
 *
 * Fills the buffer with the sequence:
 *   {start_value, start_value+1, start_value+2, ..., start_value+size-1}
 *
 * Similar to the function std::iota, wraps the function thrust::sequence
 *
 * @tparam      value_t      type of the value to operate on
 *
 * @param[in]   stream_view  stream view
 * @param[out]  d_value      device array to fill
 * @param[in]   size         number of elements in array
 * @param[in]   start_value  starting value for sequence
 *
 */
template <typename value_t>
void sequence_fill(rmm::cuda_stream_view const& stream_view,
                   value_t* d_value,
                   size_t size,
                   value_t start_value);

/**
 * @brief    Compute the maximum vertex id of an edge list
 *
 * max(d_edgelist_srcs.max(), d_edgelist_dsts.max())
 *
 * @tparam      vertex_t        vertex type
 *
 * @param[in]   stream_view     stream view
 * @param[in]   d_edgelist_srcs device array storing edge source IDs
 * @param[in]   d_edgelist_dsts device array storing edge destination IDs
 * @param[in]   num_edges       number of edges in the input source & destination arrays
 *
 * @param the maximum value occurring in the edge list
 */
template <typename vertex_t>
vertex_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                   vertex_t const* d_edgelist_srcs,
                                   vertex_t const* d_edgelist_dsts,
                                   size_t num_edges);

/**
 * @brief    Compute the maximum vertex id of an edge list
 *
 * max(d_edgelist_srcs.max(), d_edgelist_dsts.max())
 *
 * @tparam      vertex_t        vertex type
 *
 * @param[in]   stream_view     stream view
 * @param[in]   d_edgelist_srcs device array storing source IDs
 * @param[in]   d_edgelist_dsts device array storing destination IDs
 *
 * @param the maximum value occurring in the edge list
 */
template <typename vertex_t>
vertex_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                   rmm::device_uvector<vertex_t> const& d_edgelist_srcs,
                                   rmm::device_uvector<vertex_t> const& d_edgelist_dsts)
{
  return compute_maximum_vertex_id(
    stream_view, d_edgelist_srcs.data(), d_edgelist_dsts.data(), d_edgelist_srcs.size());
}

/**
 * @brief Filter zero degree vertices from this frontier
 *
 * @tparam vertex_t vertex type
 * @tparam edge_t edge type
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param d_vertices The input list of vertices
 * @param d_out_degs The output degree of each vertex
 * @return A tuple of device vectors the updated list of vertices and output degrees
 */
template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> filter_degree_0_vertices(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_vertices,
  rmm::device_uvector<edge_t>&& d_out_degs);

/**
 * @brief Check if device span is sorted
 *
 * @tparam data_t type of data in span
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param span The span of data to check
 * @return true if sorted, false if not sorted
 */
template <typename data_t>
bool is_sorted(raft::handle_t const& handle, raft::device_span<data_t> span);

/**
 * @brief Check if two device spans are equal.  Returns true if every element in the spans are
 * equal.
 *
 * @tparam data_t type of data in span
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param span1 The span of data to compare
 * @param span2 The span of data to compare
 * @return true if equal, false if not equal
 */
template <typename data_t>
bool is_equal(raft::handle_t const& handle,
              raft::device_span<data_t> span1,
              raft::device_span<data_t> span2);

/**
 * @brief Count the number of times a value appears in a span
 *
 * @tparam data_t type of data in span
 * @tparam multi_gpu True if multi-GPU computation
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param span The span of data to compare
 * @param value The value to count
 * @return The count of how many instances of that value occur
 */
template <typename data_t, bool multi_gpu>
size_t count_values(raft::handle_t const& handle,
                    raft::device_span<data_t const> span,
                    data_t value);

}  // namespace detail
}  // namespace cugraph

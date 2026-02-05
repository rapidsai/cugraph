/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

namespace cugraph {
namespace detail {

/** @defgroup utility_wrappers_cpp C++ Utility Wrappers
 */

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Fill a buffer with uniformly distributed random values
 *
 * Fills a buffer with uniformly distributed random values between
 * the specified minimum and maximum values.
 *
 * @tparam      value_t      type of the value to operate on (currently supports int32_t, int64_t,
 * float and double)
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
 * @ingroup utility_wrappers_cpp
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
 * @ingroup utility_wrappers_cpp
 * @brief    Sort a device span
 *
 * @tparam      value_t      type of the value to operate on. Must be either int32_t or int64_t.
 *
 * @param [in]  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[out]  values      device span to sort
 *
 */
template <typename value_t>
void sort_ints(raft::handle_t const& handle, raft::device_span<value_t> values);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Keep unique element from a device span
 *
 * @tparam      value_t      type of the value to operate on. Must be either int32_t or int64_t.
 *
 * @param [in]  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in]  values      device span of unique elements.
 * @return the number of unique elements.
 *
 */
template <typename value_t>
size_t unique_ints(raft::handle_t const& handle, raft::device_span<value_t> values);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Increment the values of a device span by a constant value
 *
 * @tparam      value_t      type of the value to operate on. Must be either int32_t or int64_t.
 *
 * @param[out]  values       device span to update
 * @param[in]   value        value to be added to each element of the buffer
 * @param[in]   stream_view  stream view
 *
 */
template <typename value_t>
void transform_increment_ints(raft::device_span<value_t> values,
                              value_t value,
                              rmm::cuda_stream_view const& stream_view);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Update the values of device span to 0 if it matches the compare value or 1
 *
 * @tparam      value_t      type of the value to operate on. Must be either int32_t or int64_t.
 *
 * @param[out]  values       device span with the values to compare
 * @param[out]  result       device span with the result of the comparison
 * @param[in]   compare      value to be querriedm in the values array
 * @param[in]   stream_view  stream view
 *
 */
template <typename value_t>
void transform_not_equal(raft::device_span<value_t> values,
                         raft::device_span<bool> result,
                         value_t compare,
                         rmm::cuda_stream_view const& stream_view);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Fill a buffer with a sequence of values
 *
 * Fills the buffer with the sequence:
 *   {start_value, start_value+1, start_value+2, ..., start_value+size-1}
 *
 * Similar to the function std::iota, wraps the function thrust::sequence
 *
 * @tparam      value_t      type of the value to operate on.
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
 * @ingroup utility_wrappers_cpp
 * @brief    Fill a buffer with a sequence of values with the input stride
 *
 * Fills the buffer with the sequence with the input stride:
 *   {start_value, start_value+stride, start_value+stride*2, ..., start_value+stride*(size-1)}
 *
 * @tparam      value_t      type of the value to operate on
 *
 * @param[in]   stream_view  stream view
 * @param[out]  d_value      device array to fill
 * @param[in]   size         number of elements in array
 * @param[in]   start_value  starting value for sequence
 * @param[in]   stride       input stride
 *
 */
template <typename value_t>
void stride_fill(rmm::cuda_stream_view const& stream_view,
                 value_t* d_value,
                 size_t size,
                 value_t start_value,
                 value_t stride);

/**
 * @ingroup utility_wrappers_cpp
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
 * @ingroup utility_wrappers_cpp
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
 * @ingroup utility_wrappers_cpp
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
 * @ingroup utility_wrappers_cpp
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
 * @ingroup utility_wrappers_cpp
 * @brief Count the number of times a value appears in a span
 *
 * @tparam data_t type of data in span
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param span The span of data to compare
 * @param value The value to count
 * @return The count of how many instances of that value occur
 */
template <typename data_t>
size_t count_values(raft::handle_t const& handle,
                    raft::device_span<data_t const> span,
                    data_t value);

}  // namespace detail
}  // namespace cugraph

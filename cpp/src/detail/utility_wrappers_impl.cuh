/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "c_api/array.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

namespace cugraph {
namespace detail {

template <typename value_t>
void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                         value_t* d_value,
                         size_t size,
                         value_t min_value,
                         value_t max_value,
                         raft::random::RngState& rng_state)
{
  if constexpr (std::is_integral<value_t>::value) {
    raft::random::uniformInt<value_t, size_t>(
      rng_state, d_value, size, min_value, max_value, stream_view.value());
  } else {
    raft::random::uniform<value_t, size_t>(
      rng_state, d_value, size, min_value, max_value, stream_view.value());
  }
}

template <typename value_t>
void scalar_fill(raft::handle_t const& handle, value_t* d_value, size_t size, value_t value)
{
  thrust::fill_n(handle.get_thrust_policy(), d_value, size, value);
}

template <typename value_t>
void sort_ints(raft::handle_t const& handle, raft::device_span<value_t> values)
{
  thrust::sort(handle.get_thrust_policy(), values.begin(), values.end());
}

template <typename value_t>
size_t unique_ints(raft::handle_t const& handle, raft::device_span<value_t> values)
{
  auto unique_element_last =
    thrust::unique(handle.get_thrust_policy(), values.begin(), values.end());
  return cuda::std::distance(values.begin(), unique_element_last);
}

template <typename value_t>
void sequence_fill(rmm::cuda_stream_view const& stream_view,
                   value_t* d_value,
                   size_t size,
                   value_t start_value)
{
  thrust::sequence(rmm::exec_policy(stream_view), d_value, d_value + size, start_value);
}

template <typename value_t>
void transform_increment_ints(raft::device_span<value_t> values,
                              value_t incr,
                              rmm::cuda_stream_view const& stream_view)
{
  thrust::transform(rmm::exec_policy(stream_view),
                    values.begin(),
                    values.end(),
                    values.begin(),
                    cuda::proclaim_return_type<value_t>([incr] __device__(value_t value) {
                      return static_cast<value_t>(value + incr);
                    }));
}

template <typename new_type_t>
void copy_or_transform(raft::device_span<new_type_t> output,
                       cugraph_type_erased_device_array_view_t const* input,
                       rmm::cuda_stream_view const& stream_view)
{
  auto input_ =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(input);
  if (((input_->type_ == cugraph_data_type_id_t::INT32) && (std::is_same_v<new_type_t, int32_t>)) ||
      ((input_->type_ == cugraph_data_type_id_t::INT64) && (std::is_same_v<new_type_t, int64_t>)) ||
      ((input_->type_ == cugraph_data_type_id_t::FLOAT32) && (std::is_same_v<new_type_t, float>)) ||
      ((input_->type_ == cugraph_data_type_id_t::FLOAT64) &&
       (std::is_same_v<new_type_t, double>))) {
    // dtype match so just perform a copy
    raft::copy<new_type_t>(
      output.data(), input_->as_type<new_type_t>(), input_->size_, stream_view);
  }

  else {
    // There is a dtype mismatch
    if (input_->type_ == cugraph_data_type_id_t::INT32) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input_->as_type<int32_t>(),
                        input_->as_type<int32_t>() + input_->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input_->type_ == cugraph_data_type_id_t::INT64) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input_->as_type<int64_t>(),
                        input_->as_type<int64_t>() + input_->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input_->type_ == cugraph_data_type_id_t::FLOAT32) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input_->as_type<float>(),
                        input_->as_type<float>() + input_->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input_->type_ == cugraph_data_type_id_t::FLOAT64) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input_->as_type<double>(),
                        input_->as_type<double>() + input_->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    }
  }
}

template <typename value_t>
void transform_not_equal(raft::device_span<value_t> values,
                         raft::device_span<bool> result,
                         value_t compare,
                         rmm::cuda_stream_view const& stream_view)
{
  thrust::transform(rmm::exec_policy(stream_view),
                    values.begin(),
                    values.end(),
                    result.begin(),
                    cuda::proclaim_return_type<bool>(
                      [compare] __device__(value_t value) { return compare != value; }));
}

template <typename value_t>
void stride_fill(rmm::cuda_stream_view const& stream_view,
                 value_t* d_value,
                 size_t size,
                 value_t start_value,
                 value_t stride)
{
  thrust::transform(rmm::exec_policy(stream_view),
                    thrust::make_counting_iterator(size_t{0}),
                    thrust::make_counting_iterator(size),
                    d_value,
                    cuda::proclaim_return_type<value_t>([start_value, stride] __device__(size_t i) {
                      return static_cast<value_t>(start_value + stride * i);
                    }));
}

template <typename vertex_t>
vertex_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                   vertex_t const* d_edgelist_srcs,
                                   vertex_t const* d_edgelist_dsts,
                                   size_t num_edges)
{
  auto max_v_first = thrust::make_transform_iterator(
    thrust::make_zip_iterator(d_edgelist_srcs, d_edgelist_dsts),
    cuda::proclaim_return_type<vertex_t>([] __device__(auto e) {
      return cuda::std::max(cuda::std::get<0>(e), cuda::std::get<1>(e));
    }));
  return thrust::reduce(rmm::exec_policy(stream_view),
                        max_v_first,
                        max_v_first + num_edges,
                        vertex_t{0},
                        thrust::maximum<vertex_t>{});
}

template <typename data_t>
bool is_sorted(raft::handle_t const& handle, raft::device_span<data_t> span)
{
  return thrust::is_sorted(handle.get_thrust_policy(), span.begin(), span.end());
}

template <typename data_t>
bool is_equal(raft::handle_t const& handle,
              raft::device_span<data_t> span1,
              raft::device_span<data_t> span2)
{
  return thrust::equal(handle.get_thrust_policy(), span1.begin(), span1.end(), span2.begin());
}

template <typename data_t>
size_t count_values(raft::handle_t const& handle,
                    raft::device_span<data_t const> span,
                    data_t value)
{
  return thrust::count(handle.get_thrust_policy(), span.begin(), span.end(), value);
}

}  // namespace detail
}  // namespace cugraph

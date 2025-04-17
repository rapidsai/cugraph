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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
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
#include <thrust/tuple.h>
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
  auto max_v_first =
    thrust::make_transform_iterator(thrust::make_zip_iterator(d_edgelist_srcs, d_edgelist_dsts),
                                    cuda::proclaim_return_type<vertex_t>([] __device__(auto e) {
                                      return cuda::std::max(thrust::get<0>(e), thrust::get<1>(e));
                                    }));
  return thrust::reduce(rmm::exec_policy(stream_view),
                        max_v_first,
                        max_v_first + num_edges,
                        vertex_t{0},
                        thrust::maximum<vertex_t>{});
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> filter_degree_0_vertices(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_vertices,
  rmm::device_uvector<edge_t>&& d_out_degs)
{
  auto zip_iter =
    thrust::make_zip_iterator(thrust::make_tuple(d_vertices.begin(), d_out_degs.begin()));

  CUGRAPH_EXPECTS(d_vertices.size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  "remove_if will fail, d_vertices.size() is too large");

  // FIXME: remove_if has a 32-bit overflow issue (https://github.com/NVIDIA/thrust/issues/1302)
  // Seems unlikely here so not going to work around this for now.
  auto zip_iter_end =
    thrust::remove_if(handle.get_thrust_policy(),
                      zip_iter,
                      zip_iter + d_vertices.size(),
                      zip_iter,
                      [] __device__(auto pair) { return thrust::get<1>(pair) == 0; });

  auto new_size = cuda::std::distance(zip_iter, zip_iter_end);
  d_vertices.resize(new_size, handle.get_stream());
  d_out_degs.resize(new_size, handle.get_stream());

  return std::make_tuple(std::move(d_vertices), std::move(d_out_degs));
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

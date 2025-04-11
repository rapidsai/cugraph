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

#include "detail/utility_wrappers_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                  int32_t* d_value,
                                  size_t size,
                                  int32_t min_value,
                                  int32_t max_value,
                                  raft::random::RngState& rng_state);

template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                  float* d_value,
                                  size_t size,
                                  float min_value,
                                  float max_value,
                                  raft::random::RngState& rng_state);

template void scalar_fill(raft::handle_t const& handle,
                          int32_t* d_value,
                          size_t size,
                          int32_t value);

template void scalar_fill(raft::handle_t const& handle, size_t* d_value, size_t size, size_t value);

template void scalar_fill(raft::handle_t const& handle, float* d_value, size_t size, float value);

template void sort_ints(raft::handle_t const& handle, raft::device_span<int32_t> values);

template size_t unique_ints(raft::handle_t const& handle, raft::device_span<int32_t> values);

template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                            int32_t* d_value,
                            size_t size,
                            int32_t start_value);

template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                            uint32_t* d_value,
                            size_t size,
                            uint32_t start_value);

template void transform_increment_ints(raft::device_span<int32_t> values,
                                       int32_t value,
                                       rmm::cuda_stream_view const& stream_view);

template void transform_not_equal(raft::device_span<int32_t> values,
                                  raft::device_span<bool> result,
                                  int32_t compare,
                                  rmm::cuda_stream_view const& stream_view);

template void stride_fill(rmm::cuda_stream_view const& stream_view,
                          int32_t* d_value,
                          size_t size,
                          int32_t start_value,
                          int32_t stride);

template void stride_fill(rmm::cuda_stream_view const& stream_view,
                          uint32_t* d_value,
                          size_t size,
                          uint32_t start_value,
                          uint32_t stride);

template int32_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                           int32_t const* d_edgelist_srcs,
                                           int32_t const* d_edgelist_dsts,
                                           size_t num_edges);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
filter_degree_0_vertices(raft::handle_t const& handle,
                         rmm::device_uvector<int32_t>&& d_vertices,
                         rmm::device_uvector<int32_t>&& d_out_degs);

template bool is_sorted(raft::handle_t const& handle, raft::device_span<int32_t> span);
template bool is_sorted(raft::handle_t const& handle, raft::device_span<int32_t const> span);

template bool is_equal(raft::handle_t const& handle,
                       raft::device_span<int32_t> span1,
                       raft::device_span<int32_t> span2);
template bool is_equal(raft::handle_t const& handle,
                       raft::device_span<int32_t const> span1,
                       raft::device_span<int32_t const> span2);

template size_t count_values<int32_t>(raft::handle_t const& handle,
                                      raft::device_span<int32_t const> span,
                                      int32_t value);

}  // namespace detail
}  // namespace cugraph

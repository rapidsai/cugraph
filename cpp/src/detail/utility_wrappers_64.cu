/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/utility_wrappers_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace cugraph {
namespace detail {

template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                  int64_t* d_value,
                                  size_t size,
                                  int64_t min_value,
                                  int64_t max_value,
                                  raft::random::RngState& rng_state);

template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                  double* d_value,
                                  size_t size,
                                  double min_value,
                                  double max_value,
                                  raft::random::RngState& rng_state);

template void scalar_fill(raft::handle_t const& handle,
                          int64_t* d_value,
                          size_t size,
                          int64_t value);

template void scalar_fill(raft::handle_t const& handle, double* d_value, size_t size, double value);

template void sort_ints(raft::handle_t const& handle, raft::device_span<int64_t> values);

template size_t unique_ints(raft::handle_t const& handle, raft::device_span<int64_t> values);

template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                            int64_t* d_value,
                            size_t size,
                            int64_t start_value);

template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                            uint64_t* d_value,
                            size_t size,
                            uint64_t start_value);

template void transform_increment_ints(raft::device_span<int64_t> values,
                                       int64_t value,
                                       rmm::cuda_stream_view const& stream_view);

template void transform_not_equal(raft::device_span<int64_t> values,
                                  raft::device_span<bool> result,
                                  int64_t compare,
                                  rmm::cuda_stream_view const& stream_view);

template void stride_fill(rmm::cuda_stream_view const& stream_view,
                          int64_t* d_value,
                          size_t size,
                          int64_t start_value,
                          int64_t stride);

template void stride_fill(rmm::cuda_stream_view const& stream_view,
                          uint64_t* d_value,
                          size_t size,
                          uint64_t start_value,
                          uint64_t stride);

template int64_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                           int64_t const* d_edgelist_srcs,
                                           int64_t const* d_edgelist_dsts,
                                           size_t num_edges);

template bool is_sorted(raft::handle_t const& handle, raft::device_span<int64_t> span);
template bool is_sorted(raft::handle_t const& handle, raft::device_span<int64_t const> span);

template bool is_equal(raft::handle_t const& handle,
                       raft::device_span<int64_t> span1,
                       raft::device_span<int64_t> span2);
template bool is_equal(raft::handle_t const& handle,
                       raft::device_span<int64_t const> span1,
                       raft::device_span<int64_t const> span2);

template size_t count_values<int64_t>(raft::handle_t const& handle,
                                      raft::device_span<int64_t const> span,
                                      int64_t value);

}  // namespace detail
}  // namespace cugraph

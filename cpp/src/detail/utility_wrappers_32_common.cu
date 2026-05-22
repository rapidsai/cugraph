/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace cugraph {
namespace detail {

template void uniform_random_fill(int32_t* d_value,
                                  size_t size,
                                  int32_t min_value,
                                  int32_t max_value,
                                  raft::random::RngState& rng_state,
                                  rmm::cuda_stream_view const& stream_view);

template void uniform_random_fill(float* d_value,
                                  size_t size,
                                  float min_value,
                                  float max_value,
                                  raft::random::RngState& rng_state,
                                  rmm::cuda_stream_view const& stream_view);

template void scalar_fill(int32_t* d_value,
                          size_t size,
                          int32_t value,
                          rmm::cuda_stream_view const& stream_view);

template void scalar_fill(size_t* d_value,
                          size_t size,
                          size_t value,
                          rmm::cuda_stream_view const& stream_view);

template void scalar_fill(float* d_value,
                          size_t size,
                          float value,
                          rmm::cuda_stream_view const& stream_view);

template void device_sort_impl<int32_t*>(rmm::exec_policy const& policy,
                                         int32_t* first,
                                         int32_t* last);
template void device_sort_impl<int32_t*>(rmm::exec_policy_nosync const& policy,
                                         int32_t* first,
                                         int32_t* last);

template void device_sort_impl<uint32_t*>(rmm::exec_policy const& policy,
                                          uint32_t* first,
                                          uint32_t* last);
template void device_sort_impl<uint32_t*>(rmm::exec_policy_nosync const& policy,
                                          uint32_t* first,
                                          uint32_t* last);

template size_t unique_ints(raft::device_span<int32_t> values,
                            rmm::cuda_stream_view const& stream_view);

template void sequence_fill(int32_t* d_value,
                            size_t size,
                            int32_t start_value,
                            rmm::cuda_stream_view const& stream_view);

template void sequence_fill(uint32_t* d_value,
                            size_t size,
                            uint32_t start_value,
                            rmm::cuda_stream_view const& stream_view);

template void transform_increment_ints(raft::device_span<int32_t> values,
                                       int32_t value,
                                       rmm::cuda_stream_view const& stream_view);

template void transform_not_equal(raft::device_span<int32_t> values,
                                  raft::device_span<bool> result,
                                  int32_t compare,
                                  rmm::cuda_stream_view const& stream_view);

template void stride_fill(int32_t* d_value,
                          size_t size,
                          int32_t start_value,
                          int32_t stride,
                          rmm::cuda_stream_view const& stream_view);

template void stride_fill(uint32_t* d_value,
                          size_t size,
                          uint32_t start_value,
                          uint32_t stride,
                          rmm::cuda_stream_view const& stream_view);

template int32_t compute_maximum_vertex_id(int32_t const* d_edgelist_srcs,
                                           int32_t const* d_edgelist_dsts,
                                           size_t num_edges,
                                           rmm::cuda_stream_view const& stream_view);

template bool is_sorted(raft::device_span<int32_t> span, rmm::cuda_stream_view const& stream_view);
template bool is_sorted(raft::device_span<int32_t const> span,
                        rmm::cuda_stream_view const& stream_view);

template bool is_equal(raft::device_span<int32_t> span1,
                       raft::device_span<int32_t> span2,
                       rmm::cuda_stream_view const& stream_view);
template bool is_equal(raft::device_span<int32_t const> span1,
                       raft::device_span<int32_t const> span2,
                       rmm::cuda_stream_view const& stream_view);

template size_t count_values<int32_t>(raft::device_span<int32_t const> span,
                                      int32_t value,
                                      rmm::cuda_stream_view const& stream_view);

}  // namespace detail
}  // namespace cugraph

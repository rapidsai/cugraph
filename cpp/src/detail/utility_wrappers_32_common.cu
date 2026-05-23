/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/utility_wrappers_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/export.hpp>
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

template CUGRAPH_EXPORT void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                                 int32_t* d_value,
                                                 size_t size,
                                                 int32_t min_value,
                                                 int32_t max_value,
                                                 raft::random::RngState& rng_state);

template CUGRAPH_EXPORT void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                                 float* d_value,
                                                 size_t size,
                                                 float min_value,
                                                 float max_value,
                                                 raft::random::RngState& rng_state);

template CUGRAPH_EXPORT void device_sort_impl<int32_t*>(rmm::exec_policy const& policy,
                                                        int32_t* first,
                                                        int32_t* last);
template CUGRAPH_EXPORT void device_sort_impl<int32_t*>(rmm::exec_policy_nosync const& policy,
                                                        int32_t* first,
                                                        int32_t* last);

template CUGRAPH_EXPORT void device_sort_impl<uint32_t*>(rmm::exec_policy const& policy,
                                                         uint32_t* first,
                                                         uint32_t* last);
template CUGRAPH_EXPORT void device_sort_impl<uint32_t*>(rmm::exec_policy_nosync const& policy,
                                                         uint32_t* first,
                                                         uint32_t* last);

template CUGRAPH_EXPORT void scalar_fill(raft::handle_t const& handle,
                                         int32_t* d_value,
                                         size_t size,
                                         int32_t value);

template CUGRAPH_EXPORT void scalar_fill(raft::handle_t const& handle,
                                         size_t* d_value,
                                         size_t size,
                                         size_t value);

template CUGRAPH_EXPORT void scalar_fill(raft::handle_t const& handle,
                                         float* d_value,
                                         size_t size,
                                         float value);

template CUGRAPH_EXPORT void sort_ints(raft::handle_t const& handle,
                                       raft::device_span<int32_t> values);

template CUGRAPH_EXPORT size_t unique_ints(raft::handle_t const& handle,
                                           raft::device_span<int32_t> values);

template CUGRAPH_EXPORT void sequence_fill(rmm::cuda_stream_view const& stream_view,
                                           int32_t* d_value,
                                           size_t size,
                                           int32_t start_value);

template CUGRAPH_EXPORT void sequence_fill(rmm::cuda_stream_view const& stream_view,
                                           uint32_t* d_value,
                                           size_t size,
                                           uint32_t start_value);

template CUGRAPH_EXPORT void transform_increment_ints(raft::device_span<int32_t> values,
                                                      int32_t value,
                                                      rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void transform_not_equal(raft::device_span<int32_t> values,
                                                 raft::device_span<bool> result,
                                                 int32_t compare,
                                                 rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void stride_fill(rmm::cuda_stream_view const& stream_view,
                                         int32_t* d_value,
                                         size_t size,
                                         int32_t start_value,
                                         int32_t stride);

template CUGRAPH_EXPORT void stride_fill(rmm::cuda_stream_view const& stream_view,
                                         uint32_t* d_value,
                                         size_t size,
                                         uint32_t start_value,
                                         uint32_t stride);

template CUGRAPH_EXPORT int32_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                                          int32_t const* d_edgelist_srcs,
                                                          int32_t const* d_edgelist_dsts,
                                                          size_t num_edges);

template CUGRAPH_EXPORT bool is_sorted(raft::handle_t const& handle,
                                       raft::device_span<int32_t> span);
template CUGRAPH_EXPORT bool is_sorted(raft::handle_t const& handle,
                                       raft::device_span<int32_t const> span);

template CUGRAPH_EXPORT bool is_equal(raft::handle_t const& handle,
                                      raft::device_span<int32_t> span1,
                                      raft::device_span<int32_t> span2);
template CUGRAPH_EXPORT bool is_equal(raft::handle_t const& handle,
                                      raft::device_span<int32_t const> span1,
                                      raft::device_span<int32_t const> span2);

template CUGRAPH_EXPORT size_t count_values<int32_t>(raft::handle_t const& handle,
                                                     raft::device_span<int32_t const> span,
                                                     int32_t value);

}  // namespace detail
}  // namespace cugraph

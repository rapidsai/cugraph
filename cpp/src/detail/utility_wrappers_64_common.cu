/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/utility_wrappers_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/export.hpp>

#include <raft/random/rng.cuh>

#include <rmm/exec_policy.hpp>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT void uniform_random_fill(int64_t* d_value,
                                                 size_t size,
                                                 int64_t min_value,
                                                 int64_t max_value,
                                                 raft::random::RngState& rng_state,
                                                 rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void uniform_random_fill(double* d_value,
                                                 size_t size,
                                                 double min_value,
                                                 double max_value,
                                                 raft::random::RngState& rng_state,
                                                 rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void scalar_fill(int64_t* d_value,
                                         size_t size,
                                         int64_t value,
                                         rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void scalar_fill(double* d_value,
                                         size_t size,
                                         double value,
                                         rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void device_sort_impl<int64_t*>(rmm::exec_policy const& policy,
                                                        int64_t* first,
                                                        int64_t* last);
template CUGRAPH_EXPORT void device_sort_impl<int64_t*>(rmm::exec_policy_nosync const& policy,
                                                        int64_t* first,
                                                        int64_t* last);

template CUGRAPH_EXPORT size_t unique_ints(raft::device_span<int64_t> values,
                                           rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void transform_increment_ints(raft::device_span<int64_t> values,
                                                      int64_t value,
                                                      rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void transform_not_equal(raft::device_span<int64_t> values,
                                                 raft::device_span<bool> result,
                                                 int64_t compare,
                                                 rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void sequence_fill(int64_t* d_value,
                                           size_t size,
                                           int64_t start_value,
                                           rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void sequence_fill(uint64_t* d_value,
                                           size_t size,
                                           uint64_t start_value,
                                           rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void stride_fill(int64_t* d_value,
                                         size_t size,
                                         int64_t start_value,
                                         int64_t stride,
                                         rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT void stride_fill(uint64_t* d_value,
                                         size_t size,
                                         uint64_t start_value,
                                         uint64_t stride,
                                         rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT int64_t compute_maximum_vertex_id(int64_t const* d_edgelist_srcs,
                                                          int64_t const* d_edgelist_dsts,
                                                          size_t num_edges,
                                                          rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT bool is_sorted(raft::device_span<int64_t> span,
                                       rmm::cuda_stream_view const& stream_view);
template CUGRAPH_EXPORT bool is_sorted(raft::device_span<int64_t const> span,
                                       rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT bool is_equal(raft::device_span<int64_t> span1,
                                      raft::device_span<int64_t> span2,
                                      rmm::cuda_stream_view const& stream_view);
template CUGRAPH_EXPORT bool is_equal(raft::device_span<int64_t const> span1,
                                      raft::device_span<int64_t const> span2,
                                      rmm::cuda_stream_view const& stream_view);

template CUGRAPH_EXPORT size_t count_values(raft::device_span<int64_t const> span,
                                            int64_t value,
                                            rmm::cuda_stream_view const& stream_view);

}  // namespace detail
}  // namespace cugraph

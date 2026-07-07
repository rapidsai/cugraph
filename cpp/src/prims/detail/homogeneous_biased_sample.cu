/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/prims/detail/homogeneous_biased_sample.cuh.
 */

#include "homogeneous_biased_sample_impl.cuh"

#include <cugraph/export.hpp>

#include <cstdint>

namespace cugraph {
namespace detail {

#define CUGRAPH_COMPUTE_HOMOGENEOUS_BIASED_SAMPLING_INDEX_INST(edge_t, bias_t)   \
  template CUGRAPH_EXPORT void                                                   \
  compute_homogeneous_biased_sampling_index_without_replacement<edge_t, bias_t>( \
    raft::handle_t const& handle,                                                \
    std::optional<raft::device_span<size_t const>> input_frontier_indices,       \
    raft::device_span<size_t const> input_degree_offsets,                        \
    raft::device_span<bias_t const> input_biases,                                \
    std::optional<raft::device_span<size_t const>> output_frontier_indices,      \
    raft::device_span<edge_t> output_nbr_indices,                                \
    std::optional<raft::device_span<bias_t>> output_keys,                        \
    raft::random::RngState& rng_state,                                           \
    size_t K,                                                                    \
    bool jump)

#define CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(edge_t, bias_t, multi_gpu) \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<edge_t>,                             \
                                     std::optional<rmm::device_uvector<size_t>>,              \
                                     std::vector<size_t>>                                     \
  homogeneous_biased_sample_without_replacement<edge_t, bias_t, multi_gpu>(                   \
    raft::handle_t const& handle,                                                             \
    raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,       \
    raft::host_span<size_t const> local_frontier_offsets,                                     \
    raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,               \
    raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets, \
    raft::host_span<size_t const> local_frontier_unique_key_offsets,                          \
    raft::random::RngState& rng_state,                                                        \
    size_t K)

CUGRAPH_COMPUTE_HOMOGENEOUS_BIASED_SAMPLING_INDEX_INST(std::int32_t, float);
CUGRAPH_COMPUTE_HOMOGENEOUS_BIASED_SAMPLING_INDEX_INST(std::int32_t, double);
CUGRAPH_COMPUTE_HOMOGENEOUS_BIASED_SAMPLING_INDEX_INST(std::int64_t, float);
CUGRAPH_COMPUTE_HOMOGENEOUS_BIASED_SAMPLING_INDEX_INST(std::int64_t, double);

CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int32_t, float, false);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int32_t, float, true);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int32_t, double, false);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int32_t, double, true);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int64_t, float, false);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int64_t, float, true);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int64_t, double, false);
CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST(std::int64_t, double, true);

#undef CUGRAPH_HOMOGENEOUS_BIASED_SAMPLE_WITHOUT_REPLACEMENT_INST
#undef CUGRAPH_COMPUTE_HOMOGENEOUS_BIASED_SAMPLING_INDEX_INST

}  // namespace detail
}  // namespace cugraph

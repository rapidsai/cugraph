/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/prims/detail/uniform_and_biased_replacement_sample.cuh.
 */

#include "uniform_and_biased_replacement_sample_impl.cuh"

#include <cugraph/export.hpp>
#include <cugraph/prims/detail/sample_and_compute_local_nbr_indices.cuh>

#include <cstdint>

namespace cugraph {
namespace detail {

#define CUGRAPH_COMPUTE_HOMOGENEOUS_UNIFORM_SAMPLING_INDEX_INST(edge_t)   \
  template CUGRAPH_EXPORT rmm::device_uvector<edge_t>                     \
  compute_homogeneous_uniform_sampling_index_without_replacement<edge_t>( \
    raft::handle_t const& handle,                                         \
    raft::device_span<edge_t const> frontier_degrees,                     \
    raft::random::RngState& rng_state,                                    \
    size_t K)

#define CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(edge_t, edge_type_t, bias_t, multi_gpu) \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<edge_t>,                           \
                                     std::optional<rmm::device_uvector<size_t>>,            \
                                     std::vector<size_t>>                                   \
  biased_sample_with_replacement<edge_t, edge_type_t, bias_t, multi_gpu>(                   \
    raft::handle_t const& handle,                                                           \
    raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,     \
    raft::host_span<size_t const> local_frontier_offsets,                                   \
    raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,             \
    raft::device_span<size_t const>                                                         \
      aggregate_local_frontier_unique_key_per_type_local_degree_offsets,                    \
    raft::host_span<size_t const> local_frontier_unique_key_offsets,                        \
    raft::random::RngState& rng_state,                                                      \
    raft::host_span<size_t const> Ks)

CUGRAPH_COMPUTE_HOMOGENEOUS_UNIFORM_SAMPLING_INDEX_INST(std::int32_t);
CUGRAPH_COMPUTE_HOMOGENEOUS_UNIFORM_SAMPLING_INDEX_INST(std::int64_t);

CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int32_t, std::int32_t, float, false);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int32_t, std::int32_t, float, true);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int32_t, std::int32_t, double, false);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int32_t, std::int32_t, double, true);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int64_t, std::int32_t, float, false);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int64_t, std::int32_t, float, true);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int64_t, std::int32_t, double, false);
CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST(std::int64_t, std::int32_t, double, true);

#undef CUGRAPH_BIASED_SAMPLE_WITH_REPLACEMENT_INST
#undef CUGRAPH_COMPUTE_HOMOGENEOUS_UNIFORM_SAMPLING_INDEX_INST

}  // namespace detail
}  // namespace cugraph

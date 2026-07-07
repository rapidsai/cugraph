/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/prims/detail/sampling_helpers.cuh.
 */

#include "sampling_helpers_impl.cuh"

#include <cugraph/export.hpp>

#include <cstdint>

namespace cugraph {
namespace detail {

#define CUGRAPH_COMPUTE_FRONTIER_VALUE_SUMS_INST(value_t)                                        \
  template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<value_t>, rmm::device_uvector<value_t>> \
  compute_frontier_value_sums_and_partitioned_local_value_sum_displacements<value_t>(            \
    raft::handle_t const& handle,                                                                \
    raft::device_span<value_t const> aggregate_local_frontier_local_value_sums,                  \
    raft::host_span<size_t const> local_frontier_offsets,                                        \
    size_t num_values_per_key)

#define CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HOMO_INST(edge_t, bias_t)       \
  template CUGRAPH_EXPORT void sample_nbr_index_with_replacement<edge_t, bias_t>( \
    raft::handle_t const& handle,                                                 \
    raft::device_span<edge_t const> frontier_degrees,                             \
    std::optional<raft::device_span<size_t const>> frontier_indices,              \
    raft::device_span<edge_t> nbr_indices,                                        \
    raft::random::RngState& rng_state,                                            \
    size_t K)

#define CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HETERO_INST(edge_t, edge_type_t, bias_t)     \
  template CUGRAPH_EXPORT void sample_nbr_index_with_replacement<edge_t, edge_type_t, bias_t>( \
    raft::handle_t const& handle,                                                              \
    raft::device_span<edge_t const> frontier_per_type_degrees,                                 \
    std::optional<std::tuple<raft::device_span<size_t const>,                                  \
                             raft::device_span<edge_type_t const>>> frontier_index_type_pairs, \
    raft::device_span<edge_t> per_type_nbr_indices,                                            \
    raft::random::RngState& rng_state,                                                         \
    raft::device_span<size_t const> K_offsets,                                                 \
    size_t K_sum)

#define CUGRAPH_COMPUTE_LOCAL_NBR_INDICES_FROM_PER_TYPE_INST(edge_t, edge_type_t)       \
  template CUGRAPH_EXPORT rmm::device_uvector<edge_t>                                   \
  compute_local_nbr_indices_from_per_type_local_nbr_indices<edge_t, edge_type_t>(       \
    raft::handle_t const& handle,                                                       \
    raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx, \
    raft::host_span<size_t const> local_frontier_offsets,                               \
    raft::device_span<size_t const>                                                     \
      aggregate_local_frontier_unique_key_per_type_local_degree_offsets,                \
    raft::host_span<size_t const> local_frontier_unique_key_offsets,                    \
    std::optional<std::tuple<raft::device_span<edge_type_t const>,                      \
                             raft::device_span<size_t const>>> edge_type_key_idx_pairs, \
    rmm::device_uvector<edge_t>&& per_type_local_nbr_indices,                           \
    raft::host_span<size_t const> local_frontier_sample_offsets,                        \
    raft::device_span<size_t const> K_offsets,                                          \
    size_t K_sum)

#define CUGRAPH_REMAP_LOCAL_NBR_INDICES_INST(edge_t)                                          \
  template CUGRAPH_EXPORT rmm::device_uvector<edge_t> remap_local_nbr_indices<edge_t>(        \
    raft::handle_t const& handle,                                                             \
    raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,       \
    raft::host_span<size_t const> local_frontier_offsets,                                     \
    raft::device_span<edge_t const> aggregate_local_frontier_unique_key_org_indices,          \
    raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets, \
    raft::host_span<size_t const> local_frontier_unique_key_offsets,                          \
    rmm::device_uvector<edge_t>&& local_nbr_indices,                                          \
    std::optional<raft::device_span<size_t const>> key_indices,                               \
    raft::host_span<size_t const> local_frontier_sample_offsets,                              \
    size_t K)

CUGRAPH_COMPUTE_FRONTIER_VALUE_SUMS_INST(std::int32_t);
CUGRAPH_COMPUTE_FRONTIER_VALUE_SUMS_INST(std::int64_t);

CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HOMO_INST(std::int32_t, double);
CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HOMO_INST(std::int64_t, double);

CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HETERO_INST(std::int32_t, std::int32_t, double);
CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HETERO_INST(std::int64_t, std::int32_t, double);

CUGRAPH_COMPUTE_LOCAL_NBR_INDICES_FROM_PER_TYPE_INST(std::int32_t, std::int32_t);
CUGRAPH_COMPUTE_LOCAL_NBR_INDICES_FROM_PER_TYPE_INST(std::int64_t, std::int32_t);

CUGRAPH_REMAP_LOCAL_NBR_INDICES_INST(std::int32_t);
CUGRAPH_REMAP_LOCAL_NBR_INDICES_INST(std::int64_t);

#undef CUGRAPH_REMAP_LOCAL_NBR_INDICES_INST
#undef CUGRAPH_COMPUTE_LOCAL_NBR_INDICES_FROM_PER_TYPE_INST
#undef CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HETERO_INST
#undef CUGRAPH_SAMPLE_NBR_INDEX_WITH_REPLACEMENT_HOMO_INST
#undef CUGRAPH_COMPUTE_FRONTIER_VALUE_SUMS_INST

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/prims/detail/uniform_sampling_index.cuh.
 */

#include "uniform_sampling_index_impl.cuh"

#include <cugraph/export.hpp>
#include <cugraph/prims/detail/sample_and_compute_local_nbr_indices.cuh>

#include <cstdint>

namespace cugraph {
namespace detail {

#define CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HOMO_INST(edge_t, bias_t)       \
  template CUGRAPH_EXPORT void sample_nbr_index_without_replacement<edge_t, bias_t>( \
    raft::handle_t const& handle,                                                    \
    raft::device_span<edge_t const> frontier_degrees,                                \
    std::optional<raft::device_span<size_t const>> frontier_indices,                 \
    raft::device_span<edge_t> nbr_indices,                                           \
    raft::random::RngState& rng_state,                                               \
    size_t K,                                                                        \
    bool algo_r)

#define CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HETERO_INST(edge_t, edge_type_t, bias_t)     \
  template CUGRAPH_EXPORT void sample_nbr_index_without_replacement<edge_t, edge_type_t, bias_t>( \
    raft::handle_t const& handle,                                                                 \
    raft::device_span<edge_t const> frontier_per_type_degrees,                                    \
    std::optional<std::tuple<raft::device_span<size_t const>,                                     \
                             raft::device_span<edge_type_t const>>> frontier_index_type_pairs,    \
    raft::device_span<edge_t> per_type_nbr_indices,                                               \
    raft::random::RngState& rng_state,                                                            \
    raft::device_span<size_t const> K_offsets,                                                    \
    size_t K_sum,                                                                                 \
    bool algo_r)

#define CUGRAPH_COMPUTE_HETEROGENEOUS_UNIFORM_SAMPLING_INDEX_INST(edge_t, edge_type_t)   \
  template CUGRAPH_EXPORT rmm::device_uvector<edge_t>                                    \
  compute_heterogeneous_uniform_sampling_index_without_replacement<edge_t, edge_type_t>( \
    raft::handle_t const& handle,                                                        \
    raft::device_span<edge_t const> frontier_per_type_degrees,                           \
    raft::random::RngState& rng_state,                                                   \
    raft::device_span<size_t const> K_offsets,                                           \
    size_t K_sum)

CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HOMO_INST(std::int32_t, double);
CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HOMO_INST(std::int64_t, double);

CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HETERO_INST(std::int32_t, std::int32_t, double);
CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HETERO_INST(std::int64_t, std::int32_t, double);

CUGRAPH_COMPUTE_HETEROGENEOUS_UNIFORM_SAMPLING_INDEX_INST(std::int32_t, std::int32_t);
CUGRAPH_COMPUTE_HETEROGENEOUS_UNIFORM_SAMPLING_INDEX_INST(std::int64_t, std::int32_t);

#undef CUGRAPH_COMPUTE_HETEROGENEOUS_UNIFORM_SAMPLING_INDEX_INST
#undef CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HETERO_INST
#undef CUGRAPH_SAMPLE_NBR_INDEX_WITHOUT_REPLACEMENT_HOMO_INST

}  // namespace detail
}  // namespace cugraph

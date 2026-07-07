/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for
 * cugraph/prims/detail/compact_nonzero_aggregate_local_frontier_biases.cuh.
 */

#include "compact_nonzero_aggregate_local_frontier_biases_impl.cuh"

#include <cugraph/export.hpp>

#include <cstdint>

namespace cugraph {
namespace detail {

#define CUGRAPH_COMPACT_NONZERO_AGGREGATE_LOCAL_FRONTIER_BIASES_INST(edge_t, bias_t)             \
  template CUGRAPH_EXPORT std::                                                                  \
    tuple<rmm::device_uvector<bias_t>, rmm::device_uvector<edge_t>, rmm::device_uvector<size_t>> \
    compact_nonzero_aggregate_local_frontier_biases<edge_t, bias_t>(                             \
      raft::handle_t const& handle,                                                              \
      rmm::device_uvector<bias_t>&& aggregate_local_frontier_biases,                             \
      rmm::device_uvector<size_t>&& aggregate_local_frontier_local_degree_offsets,               \
      size_t local_frontier_size,                                                                \
      bool do_expensive_check,                                                                   \
      bool multi_gpu)

CUGRAPH_COMPACT_NONZERO_AGGREGATE_LOCAL_FRONTIER_BIASES_INST(std::int32_t, float);
CUGRAPH_COMPACT_NONZERO_AGGREGATE_LOCAL_FRONTIER_BIASES_INST(std::int32_t, double);
CUGRAPH_COMPACT_NONZERO_AGGREGATE_LOCAL_FRONTIER_BIASES_INST(std::int64_t, float);
CUGRAPH_COMPACT_NONZERO_AGGREGATE_LOCAL_FRONTIER_BIASES_INST(std::int64_t, double);

#undef CUGRAPH_COMPACT_NONZERO_AGGREGATE_LOCAL_FRONTIER_BIASES_INST

}  // namespace detail
}  // namespace cugraph

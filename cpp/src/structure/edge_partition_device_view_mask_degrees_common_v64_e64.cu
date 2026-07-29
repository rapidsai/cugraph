/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The single-GPU and multi-GPU biased-sampling paths that are instantiated in
// libcugraph_common.so (see sampling/detail/sample_one_property_common_*.cu) reference the masked
// local-degree helpers.  libcugraph_common.so must be self-contained (no undefined cugraph
// symbols), so instantiate both the SG and MG variants here rather than relying on the downstream
// libcugraph.so / libcugraph_mg.so definitions.

#include "edge_partition_device_view_impl.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/export.hpp>
#include <cugraph/utilities/device_functors.cuh>

namespace cugraph {
namespace detail {

using vertex_t = int64_t;
using edge_t   = int64_t;

template CUGRAPH_EXPORT __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  cuda::std::optional<uint32_t const*> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template CUGRAPH_EXPORT __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  cuda::std::optional<uint32_t const*> edge_mask,
  raft::device_span<vertex_t const> majors,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace cugraph

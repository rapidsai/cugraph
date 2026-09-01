/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Sampling paths instantiated in libcugraph_common.so (sample_outgoing_edges_common_*,
// gather_one_hop_common_*) call masked degree / edge-count helpers through
// edge_partition_device_view_t. libcugraph_common.so must be self-contained (no undefined cugraph
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
  cuda::stream_ref stream);

template CUGRAPH_EXPORT __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  cuda::std::optional<uint32_t const*> edge_mask,
  raft::device_span<vertex_t const> majors,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  cuda::stream_ref stream);

template CUGRAPH_EXPORT __host__ void compute_number_of_edges_with_mask_async_sg(
  cuda::std::optional<uint32_t const*> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  cuda::stream_ref stream);

template CUGRAPH_EXPORT __host__ void compute_number_of_edges_with_mask_async_mg<vertex_t, edge_t>(
  cuda::std::optional<uint32_t const*> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  cuda::stream_ref stream);

}  // namespace detail
}  // namespace cugraph

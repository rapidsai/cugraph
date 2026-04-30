/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "edge_partition_device_view_impl.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/utilities/device_functors.cuh>

namespace cugraph {
namespace detail {

using vertex_t = int64_t;
using edge_t   = int64_t;

template __host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template __host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace cugraph

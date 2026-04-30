/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "edge_partition_device_view_impl.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/utilities/device_functors.cuh>

namespace cugraph {

using view_t   = edge_partition_device_view_t<int64_t, int64_t, false>;
using vertex_t = int64_t;
using edge_t   = int64_t;

// detail::compute_number_of_edges_with_mask_async_sg
template __host__ void detail::compute_number_of_edges_with_mask_async_sg<vertex_t, edge_t>(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template __host__ void detail::compute_number_of_edges_with_mask_async_sg<vertex_t, edge_t>(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template __host__ rmm::device_uvector<edge_t>
detail::compute_local_degrees_with_mask_sg<vertex_t, edge_t>(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template __host__ rmm::device_uvector<edge_t>
detail::compute_local_degrees_with_mask_sg<vertex_t, edge_t>(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

// compute_local_degrees_with_mask (MaskIterator + MajorIterator)
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<int64_t const*>(
  raft::device_span<uint32_t const>, int64_t const*, int64_t const*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<
  thrust::counting_iterator<int64_t>>(raft::device_span<uint32_t const>,
                                      thrust::counting_iterator<int64_t>,
                                      thrust::counting_iterator<int64_t>,
                                      rmm::cuda_stream_view) const;

}  // namespace cugraph

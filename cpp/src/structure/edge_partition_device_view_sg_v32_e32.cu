/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "edge_partition_device_view_impl.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/utilities/device_functors.cuh>

namespace cugraph {

using view_t = edge_partition_device_view_t<int32_t, int32_t, false>;

// detail::compute_number_of_edges_with_mask_async_sg
template __host__ void detail::compute_number_of_edges_with_mask_async_sg<int32_t, int32_t>(
  raft::device_span<uint32_t const>,
  raft::device_span<int32_t const>,
  raft::device_span<size_t>,
  raft::device_span<int32_t const>,
  rmm::cuda_stream_view);
template __host__ void detail::compute_number_of_edges_with_mask_async_sg<int32_t, int32_t>(
  raft::device_span<uint32_t const>,
  std::tuple<int32_t, int32_t>,
  raft::device_span<size_t>,
  raft::device_span<int32_t const>,
  rmm::cuda_stream_view);

template __host__ rmm::device_uvector<int32_t>
  detail::compute_local_degrees_with_mask_sg<int32_t, int32_t>(raft::device_span<uint32_t const>,
                                                               raft::device_span<int32_t const>,
                                                               raft::device_span<int32_t const>,
                                                               rmm::cuda_stream_view);

template __host__ rmm::device_uvector<int32_t>
  detail::compute_local_degrees_with_mask_sg<int32_t, int32_t>(raft::device_span<uint32_t const>,
                                                               std::tuple<int32_t, int32_t>,
                                                               raft::device_span<int32_t const>,
                                                               rmm::cuda_stream_view);

// compute_local_degrees_with_mask (MaskIterator + MajorIterator)
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<int32_t const*>(
  raft::device_span<uint32_t const>, int32_t const*, int32_t const*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<
  thrust::counting_iterator<int32_t>>(raft::device_span<uint32_t const>,
                                      thrust::counting_iterator<int32_t>,
                                      thrust::counting_iterator<int32_t>,
                                      rmm::cuda_stream_view) const;

}  // namespace cugraph

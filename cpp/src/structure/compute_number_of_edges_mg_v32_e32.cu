/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cugraph/detail/compute_number_of_edges_impl.cuh>
#include <cugraph/detail/compute_number_of_edges_functors.cuh>

namespace cugraph {

using view_t = edge_partition_device_view_t<int32_t, int32_t, true>;

// compute_number_of_edges_with_mask_async
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, int32_t*>(uint32_t const*, int32_t*, int32_t*, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, int32_t const*>(uint32_t const*, int32_t const*, int32_t const*, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, thrust::counting_iterator<int32_t>>(uint32_t const*, thrust::counting_iterator<int32_t>, thrust::counting_iterator<int32_t>, raft::device_span<size_t>, rmm::cuda_stream_view) const;

// compute_local_degrees_with_mask (MaskIterator only)
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<uint32_t const*>(uint32_t const*, rmm::cuda_stream_view) const;

// compute_local_degrees_with_mask (MaskIterator + MajorIterator)
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<uint32_t const*, int32_t*>(uint32_t const*, int32_t*, int32_t*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<uint32_t const*, int32_t const*>(uint32_t const*, int32_t const*, int32_t const*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<uint32_t const*, thrust::counting_iterator<int32_t>>(uint32_t const*, thrust::counting_iterator<int32_t>, thrust::counting_iterator<int32_t>, rmm::cuda_stream_view) const;

// bitmap iterator
using bitmap_iter_32_t = cuda::transform_iterator<detail::bitmap_offset_to_vertex_op_t<int32_t>, uint32_t const*>;
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, bitmap_iter_32_t>(uint32_t const*, bitmap_iter_32_t, bitmap_iter_32_t, raft::device_span<size_t>, rmm::cuda_stream_view) const;

// sparse-hypersparse iterator
using sh_iter_32_t = cuda::transform_iterator<detail::sparse_hypersparse_major_op_t<int32_t>, thrust::counting_iterator<int32_t>>;
template rmm::device_uvector<int32_t> view_t::compute_local_degrees_with_mask<uint32_t const*, sh_iter_32_t>(uint32_t const*, sh_iter_32_t, sh_iter_32_t, rmm::cuda_stream_view) const;

}  // namespace cugraph

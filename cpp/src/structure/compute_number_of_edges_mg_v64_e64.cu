/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cugraph/detail/compute_number_of_edges_impl.cuh>
#include <cugraph/detail/compute_number_of_edges_functors.cuh>

namespace cugraph {

using view_t = edge_partition_device_view_t<int64_t, int64_t, true>;

// compute_number_of_edges_async
template void view_t::compute_number_of_edges_async<int64_t*>(int64_t*, int64_t*, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_async<int64_t const*>(int64_t const*, int64_t const*, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_async<thrust::counting_iterator<int64_t>>(thrust::counting_iterator<int64_t>, thrust::counting_iterator<int64_t>, raft::device_span<size_t>, rmm::cuda_stream_view) const;

// compute_local_degrees (MajorIterator)
template rmm::device_uvector<int64_t> view_t::compute_local_degrees<int64_t*>(int64_t*, int64_t*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int64_t> view_t::compute_local_degrees<int64_t const*>(int64_t const*, int64_t const*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int64_t> view_t::compute_local_degrees<thrust::counting_iterator<int64_t>>(thrust::counting_iterator<int64_t>, thrust::counting_iterator<int64_t>, rmm::cuda_stream_view) const;

// compute_number_of_edges_with_mask_async
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, int64_t*>(uint32_t const*, int64_t*, int64_t*, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, int64_t const*>(uint32_t const*, int64_t const*, int64_t const*, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, thrust::counting_iterator<int64_t>>(uint32_t const*, thrust::counting_iterator<int64_t>, thrust::counting_iterator<int64_t>, raft::device_span<size_t>, rmm::cuda_stream_view) const;

// compute_local_degrees_with_mask (MaskIterator only)
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<uint32_t const*>(uint32_t const*, rmm::cuda_stream_view) const;

// compute_local_degrees_with_mask (MaskIterator + MajorIterator)
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<uint32_t const*, int64_t*>(uint32_t const*, int64_t*, int64_t*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<uint32_t const*, int64_t const*>(uint32_t const*, int64_t const*, int64_t const*, rmm::cuda_stream_view) const;
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<uint32_t const*, thrust::counting_iterator<int64_t>>(uint32_t const*, thrust::counting_iterator<int64_t>, thrust::counting_iterator<int64_t>, rmm::cuda_stream_view) const;

// bitmap iterator: cuda::transform_iterator<bitmap_offset_to_vertex_op_t<vertex_t>, uint32_t const*>
using bitmap_iter_64_t = cuda::transform_iterator<detail::bitmap_offset_to_vertex_op_t<int64_t>, uint32_t const*>;
template void view_t::compute_number_of_edges_async<bitmap_iter_64_t>(bitmap_iter_64_t, bitmap_iter_64_t, raft::device_span<size_t>, rmm::cuda_stream_view) const;
template void view_t::compute_number_of_edges_with_mask_async<uint32_t const*, bitmap_iter_64_t>(uint32_t const*, bitmap_iter_64_t, bitmap_iter_64_t, raft::device_span<size_t>, rmm::cuda_stream_view) const;

// sparse-hypersparse iterator: cuda::transform_iterator<sparse_hypersparse_major_op_t<vertex_t>, counting_iterator<vertex_t>>
using sh_iter_64_t = cuda::transform_iterator<detail::sparse_hypersparse_major_op_t<int64_t>, thrust::counting_iterator<int64_t>>;
template rmm::device_uvector<int64_t> view_t::compute_local_degrees_with_mask<uint32_t const*, sh_iter_64_t>(uint32_t const*, sh_iter_64_t, sh_iter_64_t, rmm::cuda_stream_view) const;

}  // namespace cugraph

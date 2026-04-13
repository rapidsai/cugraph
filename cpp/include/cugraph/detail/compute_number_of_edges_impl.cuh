/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_device_view.cuh>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <thrust/transform.h>

namespace cugraph {

// ============================================================================
// MG specialization: out-of-line definitions
// ============================================================================

template <typename vertex_t, typename edge_t, bool multi_gpu>
template <typename MaskIterator, typename MajorIterator>
__host__ void
edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_number_of_edges_with_mask_async(MaskIterator mask_first,
                                          MajorIterator major_first,
                                          MajorIterator major_last,
                                          raft::device_span<size_t> count,
                                          rmm::cuda_stream_view stream) const
{
  if (cuda::std::distance(major_first, major_last) == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(size_t), stream));
  }

  rmm::device_uvector<std::byte> d_tmp_storage(0, stream);
  size_t tmp_storage_bytes{0};

  if (dcs_nzd_vertices_) {
    auto local_degree_first = cuda::make_transform_iterator(
      major_first,
      detail::local_degree_op_t<vertex_t, edge_t, size_t, multi_gpu, true, MaskIterator>{
        this->offsets_,
        major_range_first_,
        *dcs_nzd_vertices_,
        *major_hypersparse_first_,
        mask_first});
    cub::DeviceReduce::Sum(static_cast<void*>(nullptr),
                           tmp_storage_bytes,
                           local_degree_first,
                           count.data(),
                           cuda::std::distance(major_first, major_last),
                           stream);
    d_tmp_storage.resize(tmp_storage_bytes, stream);
    cub::DeviceReduce::Sum(d_tmp_storage.data(),
                           tmp_storage_bytes,
                           local_degree_first,
                           count.data(),
                           cuda::std::distance(major_first, major_last),
                           stream);
  } else {
    auto local_degree_first = cuda::make_transform_iterator(
      major_first,
      detail::local_degree_op_t<vertex_t, edge_t, size_t, multi_gpu, false, MaskIterator>{
        this->offsets_,
        major_range_first_,
        std::byte{0} /* dummy */,
        std::byte{0} /* dummy */,
        mask_first});
    cub::DeviceReduce::Sum(static_cast<void*>(nullptr),
                           tmp_storage_bytes,
                           local_degree_first,
                           count.data(),
                           cuda::std::distance(major_first, major_last),
                           stream);
    d_tmp_storage.resize(tmp_storage_bytes, stream);
    cub::DeviceReduce::Sum(d_tmp_storage.data(),
                           tmp_storage_bytes,
                           local_degree_first,
                           count.data(),
                           cuda::std::distance(major_first, major_last),
                           stream);
  }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
template <typename MaskIterator>
__host__ rmm::device_uvector<edge_t>
edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_local_degrees_with_mask(MaskIterator mask_first, rmm::cuda_stream_view stream) const
{
  rmm::device_uvector<edge_t> local_degrees(this->major_range_size(), stream);
  if (dcs_nzd_vertices_) {
    assert(major_hypersparse_first_);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(this->major_range_first()),
      thrust::make_counting_iterator(this->major_range_last()),
      local_degrees.begin(),
      detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, true, MaskIterator>{
        this->offsets_,
        major_range_first_,
        *dcs_nzd_vertices_,
        major_hypersparse_first_.value_or(vertex_t{0}),
        mask_first});
  } else {
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(this->major_range_first()),
      thrust::make_counting_iterator(this->major_range_last()),
      local_degrees.begin(),
      detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false, MaskIterator>{
        this->offsets_,
        major_range_first_,
        std::byte{0} /* dummy */,
        std::byte{0} /* dummy */,
        mask_first});
  }
  return local_degrees;
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
template <typename MaskIterator, typename MajorIterator>
__host__ rmm::device_uvector<edge_t>
edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_local_degrees_with_mask(MaskIterator mask_first,
                                  MajorIterator major_first,
                                  MajorIterator major_last,
                                  rmm::cuda_stream_view stream) const
{
  rmm::device_uvector<edge_t> local_degrees(cuda::std::distance(major_first, major_last), stream);
  if (dcs_nzd_vertices_) {
    assert(major_hypersparse_first_);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      major_first,
      major_last,
      local_degrees.begin(),
      detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, true, MaskIterator>{
        this->offsets_,
        major_range_first_,
        dcs_nzd_vertices_.value(),
        major_hypersparse_first_.value_or(vertex_t{0}),
        mask_first});
  } else {
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      major_first,
      major_last,
      local_degrees.begin(),
      detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false, MaskIterator>{
        this->offsets_,
        major_range_first_,
        std::byte{0} /* dummy */,
        std::byte{0} /* dummy */,
        mask_first});
  }
  return local_degrees;
}

// ============================================================================
// SG specialization: out-of-line definitions
// ============================================================================

template <typename vertex_t, typename edge_t, bool multi_gpu>
template <typename MaskIterator, typename MajorIterator>
__host__ void
edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_number_of_edges_with_mask_async(MaskIterator mask_first,
                                          MajorIterator major_first,
                                          MajorIterator major_last,
                                          raft::device_span<size_t> count,
                                          rmm::cuda_stream_view stream) const
{
  if (cuda::std::distance(major_first, major_last) == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(size_t), stream));
    return;
  }

  rmm::device_uvector<std::byte> d_tmp_storage(0, stream);
  size_t tmp_storage_bytes{0};

  auto local_degree_first = cuda::make_transform_iterator(
    major_first,
    detail::local_degree_op_t<vertex_t, edge_t, size_t, multi_gpu, false, MaskIterator>{
      this->offsets_,
      std::byte{0} /* dummy */,
      std::byte{0} /* dummy */,
      std::byte{0} /* dummy */,
      mask_first});
  cub::DeviceReduce::Sum(static_cast<void*>(nullptr),
                         tmp_storage_bytes,
                         local_degree_first,
                         count.data(),
                         cuda::std::distance(major_first, major_last),
                         stream);
  d_tmp_storage.resize(tmp_storage_bytes, stream);
  cub::DeviceReduce::Sum(d_tmp_storage.data(),
                         tmp_storage_bytes,
                         local_degree_first,
                         count.data(),
                         cuda::std::distance(major_first, major_last),
                         stream);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
template <typename MaskIterator>
__host__ rmm::device_uvector<edge_t>
edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_local_degrees_with_mask(MaskIterator mask_first, rmm::cuda_stream_view stream) const
{
  rmm::device_uvector<edge_t> local_degrees(this->major_range_size(), stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(this->major_range_first()),
    thrust::make_counting_iterator(this->major_range_last()),
    local_degrees.begin(),
    detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false, MaskIterator>{
      this->offsets_,
      std::byte{0} /* dummy */,
      std::byte{0} /* dummy */,
      std::byte{0} /* dummy */,
      mask_first});
  return local_degrees;
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
template <typename MaskIterator, typename MajorIterator>
__host__ rmm::device_uvector<edge_t>
edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_local_degrees_with_mask(MaskIterator mask_first,
                                  MajorIterator major_first,
                                  MajorIterator major_last,
                                  rmm::cuda_stream_view stream) const
{
  rmm::device_uvector<edge_t> local_degrees(cuda::std::distance(major_first, major_last), stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    major_first,
    major_last,
    local_degrees.begin(),
    detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false, MaskIterator>{
      this->offsets_,
      std::byte{0} /* dummy */,
      std::byte{0} /* dummy */,
      std::byte{0} /* dummy */,
      mask_first});
  return local_degrees;
}

}  // namespace cugraph

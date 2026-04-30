/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_device_view.cuh>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cugraph {

namespace detail {
// ============================================================================
// MG specialization: out-of-line definitions
// ============================================================================
template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_mg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  compute_number_of_edges_with_mask_async_mg(edge_mask,
                                             majors.begin(),
                                             majors.end(),
                                             count,
                                             dcs_nzd_vertices,
                                             major_range_first,
                                             major_hypersparse_first,
                                             offsets,
                                             stream);
}

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_mg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> local_vertex_partition_range,
  raft::device_span<size_t> count,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  compute_number_of_edges_with_mask_async_mg(
    edge_mask,
    thrust::make_counting_iterator(std::get<0>(local_vertex_partition_range)),
    thrust::make_counting_iterator(std::get<1>(local_vertex_partition_range)),
    count,
    dcs_nzd_vertices,
    major_range_first,
    major_hypersparse_first,
    offsets,
    stream);
}

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  return compute_local_degrees_with_mask_mg(edge_mask,
                                            majors.begin(),
                                            majors.end(),
                                            dcs_nzd_vertices,
                                            major_range_first,
                                            major_hypersparse_first,
                                            offsets,
                                            stream);
}

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> local_vertex_partition_range,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  return compute_local_degrees_with_mask_mg(
    edge_mask,
    thrust::make_counting_iterator(std::get<0>(local_vertex_partition_range)),
    thrust::make_counting_iterator(std::get<1>(local_vertex_partition_range)),
    dcs_nzd_vertices,
    major_range_first,
    major_hypersparse_first,
    offsets,
    stream);
}

// ============================================================================
// SG specialization: out-of-line definitions
// ============================================================================

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  compute_number_of_edges_with_mask_async_sg<vertex_t, edge_t>(
    edge_mask, majors.begin(), majors.end(), count, offsets, stream);
}

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  compute_number_of_edges_with_mask_async_sg<vertex_t, edge_t>(
    edge_mask,
    thrust::make_counting_iterator(std::get<0>(vertex_partition_range)),
    thrust::make_counting_iterator(std::get<1>(vertex_partition_range)),
    count,
    offsets,
    stream);
}

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  return compute_local_degrees_with_mask_sg<vertex_t, edge_t>(
    edge_mask, majors.begin(), majors.end(), offsets, stream);
}

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  return compute_local_degrees_with_mask_sg<vertex_t, edge_t>(
    edge_mask,
    thrust::make_counting_iterator(std::get<0>(vertex_partition_range)),
    thrust::make_counting_iterator(std::get<1>(vertex_partition_range)),
    offsets,
    stream);
}

}  // namespace detail
}  // namespace cugraph

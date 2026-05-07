/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/**
 * @brief Launcher for compute_number_of_edges (single-GPU, no mask, pointer iterator).
 *
 * Wraps thrust::transform_reduce with local_degree_op_t so the CUB/Thrust kernels
 * are compiled in a single TU instead of every TU that includes edge_partition_device_view.cuh.
 * Accepts vertex_t const* as major iterator (covers the 104-instantiation kernel case).
 */
template <typename vertex_t, typename edge_t>
size_t launch_compute_number_of_edges(raft::device_span<edge_t const> offsets,
                                      vertex_t const* major_first,
                                      vertex_t const* major_last,
                                      rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_number_of_edges_async (single-GPU, no mask, pointer iterator).
 *
 * Wraps cub::DeviceReduce::Sum with local_degree_op_t.
 */
template <typename vertex_t, typename edge_t>
void launch_compute_number_of_edges_async(raft::device_span<edge_t const> offsets,
                                          vertex_t const* major_first,
                                          vertex_t const* major_last,
                                          raft::device_span<size_t> count,
                                          rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_local_degrees (single-GPU, no mask, full range).
 *
 * Wraps thrust::transform with local_degree_op_t using counting_iterator.
 */
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> launch_compute_local_degrees(raft::device_span<edge_t const> offsets,
                                                         vertex_t major_range_first,
                                                         vertex_t major_range_last,
                                                         rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_local_degrees (single-GPU, no mask, pointer iterator).
 *
 * Wraps thrust::transform with local_degree_op_t using vertex_t const* as major iterator.
 */
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> launch_compute_local_degrees(raft::device_span<edge_t const> offsets,
                                                         vertex_t const* major_first,
                                                         vertex_t const* major_last,
                                                         rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_number_of_edges_with_mask (single-GPU, pointer iterator).
 *
 * Wraps thrust::transform_reduce with local_degree_with_mask_op_t.
 */
template <typename vertex_t, typename edge_t, typename MaskIterator>
size_t launch_compute_number_of_edges_with_mask(raft::device_span<edge_t const> offsets,
                                                MaskIterator mask_first,
                                                vertex_t const* major_first,
                                                vertex_t const* major_last,
                                                rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_local_degrees_with_mask (single-GPU, full range).
 *
 * Wraps thrust::transform with local_degree_with_mask_op_t using counting_iterator.
 */
template <typename vertex_t, typename edge_t, typename MaskIterator>
rmm::device_uvector<edge_t> launch_compute_local_degrees_with_mask(
  raft::device_span<edge_t const> offsets,
  MaskIterator mask_first,
  vertex_t major_range_first,
  vertex_t major_range_last,
  rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_local_degrees_with_mask (single-GPU, pointer iterator).
 *
 * Wraps thrust::transform with local_degree_with_mask_op_t using vertex_t const* as major
 * iterator.
 */
template <typename vertex_t, typename edge_t, typename MaskIterator>
rmm::device_uvector<edge_t> launch_compute_local_degrees_with_mask(
  raft::device_span<edge_t const> offsets,
  MaskIterator mask_first,
  vertex_t const* major_first,
  vertex_t const* major_last,
  rmm::cuda_stream_view stream);

/**
 * @brief Launcher for compute_number_of_edges_with_mask_async (single-GPU, pointer iterator).
 *
 * Wraps cub::DeviceReduce::Sum with local_degree_with_mask_op_t.
 */
template <typename vertex_t, typename edge_t, typename MaskIterator>
void launch_compute_number_of_edges_with_mask_async(raft::device_span<edge_t const> offsets,
                                                    MaskIterator mask_first,
                                                    vertex_t const* major_first,
                                                    vertex_t const* major_last,
                                                    raft::device_span<size_t> count,
                                                    rmm::cuda_stream_view stream);
}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph

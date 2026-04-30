/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cassert>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t>
__device__ cuda::std::optional<vertex_t> major_hypersparse_idx_from_major_nocheck_impl(
  raft::device_span<vertex_t const> dcs_nzd_vertices, vertex_t major)
{
  // we can avoid binary search (and potentially improve performance) if we add an auxiliary array
  // or cuco::static_map (at the expense of additional memory)
  auto it =
    thrust::lower_bound(thrust::seq, dcs_nzd_vertices.begin(), dcs_nzd_vertices.end(), major);
  return it != dcs_nzd_vertices.end()
           ? (*it == major ? cuda::std::optional<vertex_t>{static_cast<vertex_t>(
                               cuda::std::distance(dcs_nzd_vertices.begin(), it))}
                           : cuda::std::nullopt)
           : cuda::std::nullopt;
}

template <typename vertex_t,
          typename edge_t,
          typename return_type_t,
          bool multi_gpu,
          bool use_dcs,
          typename MaskIterator = uint32_t const*>
struct local_degree_op_t {
  raft::device_span<edge_t const> offsets{};
  std::conditional_t<multi_gpu, vertex_t, std::byte /* dummy */> major_range_first{};

  std::conditional_t<use_dcs, raft::device_span<vertex_t const>, std::byte /* dummy */>
    dcs_nzd_vertices{};
  std::conditional_t<use_dcs, vertex_t, std::byte /* dummy */> major_hypersparse_first{};

  MaskIterator mask_first{};

  __device__ return_type_t compute_degree(edge_t offset, edge_t degree) const
  {
    if (mask_first) {
      return static_cast<return_type_t>(count_set_bits(mask_first, offset, degree));
    } else {
      return static_cast<return_type_t>(degree);
    }
  }

  __device__ return_type_t operator()(vertex_t major) const
  {
    if constexpr (multi_gpu) {
      vertex_t idx{};
      if constexpr (use_dcs) {
        if (major < major_hypersparse_first) {
          idx = major - major_range_first;
          return compute_degree(offsets[idx], offsets[idx + 1] - offsets[idx]);
        } else {
          auto major_hypersparse_idx =
            major_hypersparse_idx_from_major_nocheck_impl(dcs_nzd_vertices, major);
          if (major_hypersparse_idx) {
            idx = (major_hypersparse_first - major_range_first) + *major_hypersparse_idx;
            return compute_degree(offsets[idx], offsets[idx + 1] - offsets[idx]);
          } else {
            return return_type_t{0};
          }
        }
      } else {
        idx = major - major_range_first;
        return compute_degree(offsets[idx], offsets[idx + 1] - offsets[idx]);
      }
    } else {
      return compute_degree(offsets[major], offsets[major + 1] - offsets[major]);
    }
  }
};

template <typename vertex_t, typename edge_t>
class edge_partition_device_view_base_t {
 public:
  edge_partition_device_view_base_t(raft::device_span<edge_t const> offsets,
                                    raft::device_span<vertex_t const> indices)
    : offsets_(offsets), indices_(indices)
  {
  }

  __host__ __device__ edge_t number_of_edges() const
  {
    return static_cast<edge_t>(indices_.size());
  }

  __host__ __device__ edge_t const* offsets() const { return offsets_.data(); }
  __host__ __device__ vertex_t const* indices() const { return indices_.data(); }

  __device__ vertex_t major_idx_from_local_edge_idx_nocheck(edge_t local_edge_idx) const noexcept
  {
    return static_cast<vertex_t>(cuda::std::distance(
      offsets_.begin() + 1,
      thrust::upper_bound(thrust::seq, offsets_.begin() + 1, offsets_.end(), local_edge_idx)));
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ cuda::std::tuple<vertex_t const*, edge_t, edge_t> local_edges(
    vertex_t major_idx) const noexcept
  {
    auto edge_offset  = offsets_[major_idx];
    auto local_degree = offsets_[major_idx + 1] - edge_offset;
    auto indices      = indices_.data() + edge_offset;
    return cuda::std::make_tuple(indices, edge_offset, local_degree);
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ edge_t local_degree(vertex_t major_idx) const noexcept
  {
    return offsets_[major_idx + 1] - offsets_[major_idx];
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ edge_t local_offset(vertex_t major_idx) const noexcept { return offsets_[major_idx]; }

 protected:
  // should be trivially copyable to device
  raft::device_span<edge_t const> offsets_{nullptr};
  raft::device_span<vertex_t const> indices_{nullptr};
};

template <typename vertex_t, typename edge_t, typename MajorIterator>
__host__ void compute_number_of_edges_with_mask_async_mg(
  raft::device_span<uint32_t const> edge_mask,
  MajorIterator major_first,
  MajorIterator major_last,
  raft::device_span<size_t> count,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  if (cuda::std::distance(major_first, major_last) == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(size_t), stream));
    return;
  }

  rmm::device_uvector<std::byte> d_tmp_storage(0, stream);
  size_t tmp_storage_bytes{0};

  if (dcs_nzd_vertices) {
    assert(major_hypersparse_first);
    auto local_degree_first = cuda::make_transform_iterator(
      major_first,
      detail::local_degree_op_t<vertex_t, edge_t, size_t, true, true>{
        offsets, major_range_first, *dcs_nzd_vertices, *major_hypersparse_first, edge_mask.data()});
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
      detail::local_degree_op_t<vertex_t, edge_t, size_t, true, false>{offsets,
                                                                       major_range_first,
                                                                       std::byte{0} /* dummy */,
                                                                       std::byte{0} /* dummy */,
                                                                       edge_mask.data()});
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

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_mg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_mg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> local_vertex_partition_range,
  raft::device_span<size_t> count,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t, typename MajorIterator>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  raft::device_span<uint32_t const> edge_mask,
  MajorIterator major_first,
  MajorIterator major_last,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<edge_t> local_degrees(cuda::std::distance(major_first, major_last), stream);
  if (dcs_nzd_vertices) {
    assert(major_hypersparse_first);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      major_first,
      major_last,
      local_degrees.begin(),
      detail::local_degree_op_t<vertex_t, edge_t, edge_t, true, true>{
        offsets, major_range_first, *dcs_nzd_vertices, *major_hypersparse_first, edge_mask.data()});
  } else {
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      major_first,
      major_last,
      local_degrees.begin(),
      detail::local_degree_op_t<vertex_t, edge_t, edge_t, true, false>{offsets,
                                                                       major_range_first,
                                                                       std::byte{0} /* dummy */,
                                                                       std::byte{0} /* dummy */,
                                                                       edge_mask.data()});
  }
  return local_degrees;
}

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_mg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> local_vertex_partition_range,
  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
  vertex_t major_range_first,
  cuda::std::optional<vertex_t> major_hypersparse_first,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t, typename MajorIterator>
__host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  MajorIterator major_first,
  MajorIterator major_last,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  if (cuda::std::distance(major_first, major_last) == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(count.data(), 0, sizeof(size_t), stream));
    return;
  }

  rmm::device_uvector<std::byte> d_tmp_storage(0, stream);
  size_t tmp_storage_bytes{0};

  auto local_degree_first = cuda::make_transform_iterator(
    major_first,
    detail::local_degree_op_t<vertex_t, edge_t, size_t, false, false>{offsets,
                                                                      std::byte{0} /* dummy */,
                                                                      std::byte{0} /* dummy */,
                                                                      std::byte{0} /* dummy */,
                                                                      edge_mask.data()});
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

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t>
__host__ void compute_number_of_edges_with_mask_async_sg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<size_t> count,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t, typename MajorIterator>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  MajorIterator major_first,
  MajorIterator major_last,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<edge_t> local_degrees(cuda::std::distance(major_first, major_last), stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    major_first,
    major_last,
    local_degrees.begin(),
    detail::local_degree_op_t<vertex_t, edge_t, edge_t, false, false>{offsets,
                                                                      std::byte{0} /* dummy */,
                                                                      std::byte{0} /* dummy */,
                                                                      std::byte{0} /* dummy */,
                                                                      edge_mask.data()});
  return local_degrees;
}

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  raft::device_span<vertex_t const> majors,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

template <typename vertex_t, typename edge_t>
__host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask_sg(
  raft::device_span<uint32_t const> edge_mask,
  std::tuple<vertex_t, vertex_t> vertex_partition_range,
  raft::device_span<edge_t const> offsets,
  rmm::cuda_stream_view stream);

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
class edge_partition_device_view_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
class edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::edge_partition_device_view_base_t<vertex_t, edge_t> {
 public:
  edge_partition_device_view_t(edge_partition_view_t<vertex_t, edge_t, multi_gpu> view)
    : detail::edge_partition_device_view_base_t<vertex_t, edge_t>(view.offsets(), view.indices()),
      dcs_nzd_vertices_(detail::to_thrust_optional(view.dcs_nzd_vertices())),
      dcs_nzd_range_bitmap_(detail::to_thrust_optional(view.dcs_nzd_range_bitmap())),
      major_hypersparse_first_(detail::to_thrust_optional(view.major_hypersparse_first())),
      major_range_first_(view.major_range_first()),
      major_range_last_(view.major_range_last()),
      minor_range_first_(view.minor_range_first()),
      minor_range_last_(view.minor_range_last()),
      major_value_start_offset_(view.major_value_start_offset())
  {
  }

  template <typename T>
  __host__ void compute_number_of_edges_with_mask_async(raft::device_span<uint32_t const> edge_mask,
                                                        raft::device_span<T const> majors,
                                                        raft::device_span<size_t> count,
                                                        rmm::cuda_stream_view stream) const
  {
    detail::compute_number_of_edges_with_mask_async_mg(edge_mask,
                                                       majors,
                                                       count,
                                                       dcs_nzd_vertices_,
                                                       major_range_first_,
                                                       major_hypersparse_first_,
                                                       this->offsets_,
                                                       stream);
  }

  __host__ void compute_number_of_edges_with_mask_async(
    raft::device_span<uint32_t const> edge_mask,
    std::tuple<vertex_t, vertex_t> vertex_partition_range,
    raft::device_span<size_t> count,
    rmm::cuda_stream_view stream) const
  {
    detail::compute_number_of_edges_with_mask_async_mg(edge_mask,
                                                       vertex_partition_range,
                                                       count,
                                                       dcs_nzd_vertices_,
                                                       major_range_first_,
                                                       major_hypersparse_first_,
                                                       this->offsets_,
                                                       stream);
  }

  template <typename MajorIterator>
  __host__ void compute_number_of_edges_with_mask_async(raft::device_span<uint32_t const> edge_mask,
                                                        MajorIterator major_first,
                                                        MajorIterator major_last,
                                                        raft::device_span<size_t> count,
                                                        rmm::cuda_stream_view stream) const
  {
    detail::compute_number_of_edges_with_mask_async_mg(edge_mask,
                                                       major_first,
                                                       major_last,
                                                       count,
                                                       dcs_nzd_vertices_,
                                                       major_range_first_,
                                                       major_hypersparse_first_,
                                                       this->offsets_,
                                                       stream);
  }

  template <typename T>
  __host__ void compute_number_of_edges_async(raft::device_span<T const> majors,
                                              raft::device_span<size_t> count,
                                              rmm::cuda_stream_view stream) const
  {
    compute_number_of_edges_with_mask_async({}, majors, count, stream);
  }

  __host__ void compute_number_of_edges_async(std::tuple<vertex_t, vertex_t> vertex_partition_range,
                                              raft::device_span<size_t> count,
                                              rmm::cuda_stream_view stream) const
  {
    compute_number_of_edges_with_mask_async({}, vertex_partition_range, count, stream);
  }

  template <typename MajorIterator>
  __host__ void compute_number_of_edges_async(MajorIterator major_first,
                                              MajorIterator major_last,
                                              raft::device_span<size_t> count,
                                              rmm::cuda_stream_view stream) const
  {
    compute_number_of_edges_with_mask_async({}, major_first, major_last, count, stream);
  }

  template <typename T>
  __host__ size_t compute_number_of_edges_with_mask(raft::device_span<uint32_t const> edge_mask,
                                                    raft::device_span<T const> majors,
                                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_with_mask_async(
      edge_mask, majors, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  __host__ size_t
  compute_number_of_edges_with_mask(raft::device_span<uint32_t const> edge_mask,
                                    std::tuple<vertex_t, vertex_t> vertex_partition_range,
                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_with_mask_async(
      edge_mask, vertex_partition_range, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  template <typename MajorIterator>
  __host__ size_t compute_number_of_edges_with_mask(raft::device_span<uint32_t const> edge_mask,
                                                    MajorIterator major_first,
                                                    MajorIterator major_last,
                                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_with_mask_async(
      edge_mask, major_first, major_last, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  template <typename T>
  __host__ size_t compute_number_of_edges(raft::device_span<T const> majors,
                                          rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_async(majors, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  __host__ size_t compute_number_of_edges(std::tuple<vertex_t, vertex_t> vertex_partition_range,
                                          rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_async(
      vertex_partition_range, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  template <typename MajorIterator>
  __host__ size_t compute_number_of_edges(MajorIterator major_first,
                                          MajorIterator major_last,
                                          rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_async(
      major_first, major_last, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask, rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_mg(
      edge_mask,
      std::tuple<vertex_t, vertex_t>{this->major_range_first(), this->major_range_last()},
      dcs_nzd_vertices_,
      major_range_first_,
      major_hypersparse_first_,
      this->offsets_,
      stream);
  }

  template <typename T>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    raft::device_span<T const> majors,
    rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_mg(edge_mask,
                                                      majors,
                                                      dcs_nzd_vertices_,
                                                      major_range_first_,
                                                      major_hypersparse_first_,
                                                      this->offsets_,
                                                      stream);
  }

  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    std::tuple<vertex_t, vertex_t> vertex_partition_range,
    rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_mg(edge_mask,
                                                      vertex_partition_range,
                                                      dcs_nzd_vertices_,
                                                      major_range_first_,
                                                      major_hypersparse_first_,
                                                      this->offsets_,
                                                      stream);
  }

  template <typename MajorIterator>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    MajorIterator major_first,
    MajorIterator major_last,
    rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_mg(edge_mask,
                                                      major_first,
                                                      major_last,
                                                      dcs_nzd_vertices_,
                                                      major_range_first_,
                                                      major_hypersparse_first_,
                                                      this->offsets_,
                                                      stream);
  }

  template <typename T, std::enable_if_t<!std::is_const_v<T>, int> = 0>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    T* major_first,
    T* major_last,
    rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask(
      edge_mask, static_cast<T const*>(major_first), static_cast<T const*>(major_last), stream);
  }

  template <typename T>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees(raft::device_span<T const> majors,
                                                             rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask({}, majors, stream);
  }

  __host__ rmm::device_uvector<edge_t> compute_local_degrees(
    std::tuple<vertex_t, vertex_t> vertex_partition_range, rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask({}, vertex_partition_range, stream);
  }

  template <typename MajorIterator>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees(MajorIterator major_first,
                                                             MajorIterator major_last,
                                                             rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask({}, major_first, major_last, stream);
  }

  __host__ __device__ vertex_t major_value_start_offset() const
  {
    return major_value_start_offset_;
  }

  __host__ __device__ cuda::std::optional<vertex_t> major_hypersparse_first() const noexcept
  {
    return major_hypersparse_first_;
  }

  __host__ __device__ vertex_t major_range_first() const noexcept { return major_range_first_; }

  __host__ __device__ vertex_t major_range_last() const noexcept { return major_range_last_; }

  __host__ __device__ vertex_t major_range_size() const noexcept
  {
    return major_range_last_ - major_range_first_;
  }

  __host__ __device__ vertex_t minor_range_first() const noexcept { return minor_range_first_; }

  __host__ __device__ vertex_t minor_range_last() const noexcept { return minor_range_last_; }

  __host__ __device__ vertex_t minor_range_size() const noexcept
  {
    return minor_range_last_ - minor_range_first_;
  }

  __host__ __device__ vertex_t major_offset_from_major_nocheck(vertex_t major) const noexcept
  {
    return major - major_range_first_;
  }

  __host__ __device__ vertex_t minor_offset_from_minor_nocheck(vertex_t minor) const noexcept
  {
    return minor - minor_range_first_;
  }

  __host__ __device__ vertex_t major_from_major_offset_nocheck(vertex_t major_offset) const noexcept
  {
    return major_range_first_ + major_offset;
  }

  __device__ cuda::std::optional<vertex_t> major_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    if (major_hypersparse_first_ && (major >= *major_hypersparse_first_)) {
      auto major_hypersparse_idx =
        detail::major_hypersparse_idx_from_major_nocheck_impl(*dcs_nzd_vertices_, major);
      return major_hypersparse_idx
               ? cuda::std::make_optional((*major_hypersparse_first_ - major_range_first_) +
                                          *major_hypersparse_idx)
               : cuda::std::nullopt;
    } else {
      return major - major_range_first_;
    }
  }

  __device__ vertex_t major_from_major_idx_nocheck(vertex_t major_idx) const noexcept
  {
    if (major_hypersparse_first_) {
      return major_idx >= (*major_hypersparse_first_ - major_range_first_)
               ? (*dcs_nzd_vertices_)[major_idx - (*major_hypersparse_first_ - major_range_first_)]
               : major_from_major_offset_nocheck(major_idx);
    } else {  // major_idx == major_offset
      return major_from_major_offset_nocheck(major_idx);
    }
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __device__ cuda::std::optional<vertex_t> major_hypersparse_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    if (dcs_nzd_vertices_) {
      return detail::major_hypersparse_idx_from_major_nocheck_impl(*dcs_nzd_vertices_, major);
    } else {
      return cuda::std::nullopt;
    }
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __device__ cuda::std::optional<vertex_t> major_from_major_hypersparse_idx_nocheck(
    vertex_t major_hypersparse_idx) const noexcept
  {
    return dcs_nzd_vertices_
             ? cuda::std::optional<vertex_t>{(*dcs_nzd_vertices_)[major_hypersparse_idx]}
             : cuda::std::nullopt;
  }

  __host__ __device__ vertex_t minor_from_minor_offset_nocheck(vertex_t minor_offset) const noexcept
  {
    return minor_range_first_ + minor_offset;
  }

  // FIXME: better return cuda::std::optional<raft::device_span<vertex_t const>> for consistency
  // (see dcs_nzd_range_bitmap())
  __host__ __device__ cuda::std::optional<vertex_t const*> dcs_nzd_vertices() const
  {
    return dcs_nzd_vertices_ ? cuda::std::optional<vertex_t const*>{(*dcs_nzd_vertices_).data()}
                             : cuda::std::nullopt;
  }

  __host__ __device__ cuda::std::optional<vertex_t> dcs_nzd_vertex_count() const
  {
    return dcs_nzd_vertices_
             ? cuda::std::optional<vertex_t>{static_cast<vertex_t>((*dcs_nzd_vertices_).size())}
             : cuda::std::nullopt;
  }

  __host__ __device__ cuda::std::optional<raft::device_span<uint32_t const>> dcs_nzd_range_bitmap()
    const
  {
    return dcs_nzd_range_bitmap_
             ? cuda::std::make_optional<raft::device_span<uint32_t const>>(
                 (*dcs_nzd_range_bitmap_).data(), (*dcs_nzd_range_bitmap_).size())
             : cuda::std::nullopt;
  }

 private:
  // should be trivially copyable to device

  cuda::std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices_{cuda::std::nullopt};
  cuda::std::optional<raft::device_span<uint32_t const>> dcs_nzd_range_bitmap_{cuda::std::nullopt};
  cuda::std::optional<vertex_t> major_hypersparse_first_{cuda::std::nullopt};

  vertex_t major_range_first_{0};
  vertex_t major_range_last_{0};
  vertex_t minor_range_first_{0};
  vertex_t minor_range_last_{0};

  vertex_t major_value_start_offset_{0};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
class edge_partition_device_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::edge_partition_device_view_base_t<vertex_t, edge_t> {
 public:
  edge_partition_device_view_t(edge_partition_view_t<vertex_t, edge_t, multi_gpu> view)
    : detail::edge_partition_device_view_base_t<vertex_t, edge_t>(view.offsets(), view.indices()),
      number_of_vertices_(view.major_range_last())
  {
  }

  template <typename T>
  __host__ void compute_number_of_edges_with_mask_async(raft::device_span<uint32_t const> edge_mask,
                                                        raft::device_span<T const> majors,
                                                        raft::device_span<size_t> count,
                                                        rmm::cuda_stream_view stream) const
  {
    detail::compute_number_of_edges_with_mask_async_sg(
      edge_mask, majors, count, this->offsets_, stream);
  }

  __host__ void compute_number_of_edges_with_mask_async(
    raft::device_span<uint32_t const> edge_mask,
    std::tuple<vertex_t, vertex_t> vertex_partition_range,
    raft::device_span<size_t> count,
    rmm::cuda_stream_view stream) const
  {
    detail::compute_number_of_edges_with_mask_async_sg(
      edge_mask, vertex_partition_range, count, this->offsets_, stream);
  }

  template <typename MajorIterator>
  __host__ void compute_number_of_edges_with_mask_async(raft::device_span<uint32_t const> edge_mask,
                                                        MajorIterator major_first,
                                                        MajorIterator major_last,
                                                        raft::device_span<size_t> count,
                                                        rmm::cuda_stream_view stream) const
  {
    detail::compute_number_of_edges_with_mask_async_sg<vertex_t, edge_t>(
      edge_mask, major_first, major_last, count, this->offsets_, stream);
  }

  template <typename T>
  __host__ void compute_number_of_edges_async(raft::device_span<T const> majors,
                                              raft::device_span<size_t> count,
                                              rmm::cuda_stream_view stream) const
  {
    compute_number_of_edges_with_mask_async({}, majors, count, stream);
  }

  __host__ void compute_number_of_edges_async(std::tuple<vertex_t, vertex_t> vertex_partition_range,
                                              raft::device_span<size_t> count,
                                              rmm::cuda_stream_view stream) const
  {
    compute_number_of_edges_with_mask_async({}, vertex_partition_range, count, stream);
  }

  template <typename MajorIterator>
  __host__ void compute_number_of_edges_async(MajorIterator major_first,
                                              MajorIterator major_last,
                                              raft::device_span<size_t> count,
                                              rmm::cuda_stream_view stream) const
  {
    compute_number_of_edges_with_mask_async({}, major_first, major_last, count, stream);
  }

  template <typename T>
  __host__ size_t compute_number_of_edges_with_mask(raft::device_span<uint32_t const> edge_mask,
                                                    raft::device_span<T const> majors,
                                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_with_mask_async(
      edge_mask, majors, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  __host__ size_t
  compute_number_of_edges_with_mask(raft::device_span<uint32_t const> edge_mask,
                                    std::tuple<vertex_t, vertex_t> vertex_partition_range,
                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_with_mask_async(
      edge_mask, vertex_partition_range, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  template <typename MajorIterator>
  __host__ size_t compute_number_of_edges_with_mask(raft::device_span<uint32_t const> edge_mask,
                                                    MajorIterator major_first,
                                                    MajorIterator major_last,
                                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_with_mask_async(
      edge_mask, major_first, major_last, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  template <typename T>
  __host__ size_t compute_number_of_edges(raft::device_span<T const> majors,
                                          rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_async(majors, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  __host__ size_t compute_number_of_edges(std::tuple<vertex_t, vertex_t> vertex_partition_range,
                                          rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_async(
      vertex_partition_range, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  template <typename MajorIterator>
  __host__ size_t compute_number_of_edges(MajorIterator major_first,
                                          MajorIterator major_last,
                                          rmm::cuda_stream_view stream) const
  {
    rmm::device_scalar<size_t> count(size_t{0}, stream);
    compute_number_of_edges_async(
      major_first, major_last, raft::device_span<size_t>{count.data(), 1}, stream);
    return count.value(stream);
  }

  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask, rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_sg(
      edge_mask,
      std::tuple<vertex_t, vertex_t>{this->major_range_first(), this->major_range_last()},
      this->offsets_,
      stream);
  }

  template <typename T>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    raft::device_span<T const> majors,
    rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_sg(edge_mask, majors, this->offsets_, stream);
  }

  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    std::tuple<vertex_t, vertex_t> vertex_partition_range,
    rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_sg(
      edge_mask, vertex_partition_range, this->offsets_, stream);
  }

  template <typename MajorIterator>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    MajorIterator major_first,
    MajorIterator major_last,
    rmm::cuda_stream_view stream) const
  {
    return detail::compute_local_degrees_with_mask_sg<vertex_t, edge_t>(
      edge_mask, major_first, major_last, this->offsets_, stream);
  }

  template <typename T, std::enable_if_t<!std::is_const_v<T>, int> = 0>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees_with_mask(
    raft::device_span<uint32_t const> edge_mask,
    T* major_first,
    T* major_last,
    rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask(
      edge_mask, static_cast<T const*>(major_first), static_cast<T const*>(major_last), stream);
  }

  template <typename T>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees(raft::device_span<T const> majors,
                                                             rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask({}, majors, stream);
  }

  __host__ rmm::device_uvector<edge_t> compute_local_degrees(
    std::tuple<vertex_t, vertex_t> vertex_partition_range, rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask({}, vertex_partition_range, stream);
  }

  template <typename MajorIterator>
  __host__ rmm::device_uvector<edge_t> compute_local_degrees(MajorIterator major_first,
                                                             MajorIterator major_last,
                                                             rmm::cuda_stream_view stream) const
  {
    return compute_local_degrees_with_mask({}, major_first, major_last, stream);
  }

  __host__ __device__ vertex_t major_value_start_offset() const { return vertex_t{0}; }

  __host__ __device__ cuda::std::optional<vertex_t> major_hypersparse_first() const noexcept
  {
    assert(false);
    return cuda::std::nullopt;
  }

  __host__ __device__ constexpr vertex_t major_range_first() const noexcept { return vertex_t{0}; }

  __host__ __device__ vertex_t major_range_last() const noexcept { return number_of_vertices_; }

  __host__ __device__ vertex_t major_range_size() const noexcept { return number_of_vertices_; }

  __host__ __device__ constexpr vertex_t minor_range_first() const noexcept { return vertex_t{0}; }

  __host__ __device__ vertex_t minor_range_last() const noexcept { return number_of_vertices_; }

  __host__ __device__ vertex_t minor_range_size() const noexcept { return number_of_vertices_; }

  __host__ __device__ vertex_t major_offset_from_major_nocheck(vertex_t major) const noexcept
  {
    return major;
  }

  __host__ __device__ vertex_t minor_offset_from_minor_nocheck(vertex_t minor) const noexcept
  {
    return minor;
  }

  __host__ __device__ vertex_t major_from_major_offset_nocheck(vertex_t major_offset) const noexcept
  {
    return major_offset;
  }

  __device__ cuda::std::optional<vertex_t> major_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    return major_offset_from_major_nocheck(major);
  }

  __device__ vertex_t major_from_major_idx_nocheck(vertex_t major_idx) const noexcept
  {
    return major_from_major_offset_nocheck(major_idx);
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __device__ cuda::std::optional<vertex_t> major_hypersparse_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    assert(false);
    return cuda::std::nullopt;
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __device__ cuda::std::optional<vertex_t> major_from_major_hypersparse_idx_nocheck(
    vertex_t major_hypersparse_idx) const noexcept
  {
    assert(false);
    return cuda::std::nullopt;
  }

  __host__ __device__ vertex_t minor_from_minor_offset_nocheck(vertex_t minor_offset) const noexcept
  {
    return minor_offset;
  }

  __host__ __device__ cuda::std::optional<vertex_t const*> dcs_nzd_vertices() const
  {
    return cuda::std::nullopt;
  }

  __host__ __device__ cuda::std::optional<vertex_t> dcs_nzd_vertex_count() const
  {
    return cuda::std::nullopt;
  }

 private:
  vertex_t number_of_vertices_{};
};

namespace detail {

// Maps a linear major index to either a sparse-range vertex or a hypersparse DCS vertex.
// Used in per_v_transform_reduce_dst_key_aggregated_outgoing_e.
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct sparse_hypersparse_major_op_t {
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition;
  __device__ vertex_t operator()(vertex_t i) const
  {
    return edge_partition.major_from_major_idx_nocheck(i);
  }
};

}  // namespace detail

}  // namespace cugraph

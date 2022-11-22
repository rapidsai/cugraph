/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cugraph/edge_partition_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/device_span.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cassert>
#include <optional>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t>
__device__ thrust::optional<vertex_t> major_hypersparse_idx_from_major_nocheck_impl(
  raft::device_span<vertex_t const> dcs_nzd_vertices, vertex_t major)
{
  // we can avoid binary search (and potentially improve performance) if we add an auxiliary array
  // or cuco::static_map (at the expense of additional memory)
  auto it =
    thrust::lower_bound(thrust::seq, dcs_nzd_vertices.begin(), dcs_nzd_vertices.end(), major);
  return it != dcs_nzd_vertices.end()
           ? (*it == major ? thrust::optional<vertex_t>{static_cast<vertex_t>(
                               thrust::distance(dcs_nzd_vertices.begin(), it))}
                           : thrust::nullopt)
           : thrust::nullopt;
}

template <typename vertex_t, typename edge_t, typename return_type_t, bool multi_gpu, bool use_dcs>
struct local_degree_op_t {
  raft::device_span<edge_t const> offsets{};
  std::conditional_t<multi_gpu, vertex_t, std::byte /* dummy */> major_range_first{};

  std::conditional_t<use_dcs, raft::device_span<vertex_t const>, std::byte /* dummy */>
    dcs_nzd_vertices{};
  std::conditional_t<use_dcs, vertex_t, std::byte /* dummy */> major_hypersparse_first{};

  __device__ return_type_t operator()(vertex_t major) const
  {
    if constexpr (multi_gpu) {
      vertex_t idx{};
      if constexpr (use_dcs) {
        if (major < major_hypersparse_first) {
          idx = major - major_range_first;
          return static_cast<return_type_t>(offsets[idx + 1] - offsets[idx]);
        } else {
          auto major_hypersparse_idx =
            major_hypersparse_idx_from_major_nocheck_impl(dcs_nzd_vertices, major);
          if (major_hypersparse_idx) {
            idx = (major_hypersparse_first - major_range_first) + *major_hypersparse_idx;
            return static_cast<return_type_t>(offsets[idx + 1] - offsets[idx]);
          } else {
            return return_type_t{0};
          }
        }
      } else {
        idx = major - major_range_first;
        return static_cast<return_type_t>(offsets[idx + 1] - offsets[idx]);
      }
    } else {
      return static_cast<return_type_t>(offsets[major + 1] - offsets[major]);
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

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ thrust::tuple<vertex_t const*, edge_t, edge_t> local_edges(
    vertex_t major_idx) const noexcept
  {
    auto edge_offset  = offsets_[major_idx];
    auto local_degree = offsets_[major_idx + 1] - edge_offset;
    auto indices      = indices_.data() + edge_offset;
    return thrust::make_tuple(indices, edge_offset, local_degree);
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
      major_hypersparse_first_(detail::to_thrust_optional(view.major_hypersparse_first())),
      major_range_first_(view.major_range_first()),
      major_range_last_(view.major_range_last()),
      minor_range_first_(view.minor_range_first()),
      minor_range_last_(view.minor_range_last()),
      major_value_start_offset_(view.major_value_start_offset())
  {
  }

  size_t compute_number_of_edges(raft::device_span<vertex_t const> majors,
                                 rmm::cuda_stream_view stream) const
  {
    return dcs_nzd_vertices_ ? thrust::transform_reduce(
                                 rmm::exec_policy(stream),
                                 majors.begin(),
                                 majors.end(),
                                 detail::local_degree_op_t<
                                   vertex_t,
                                   edge_t,
                                   size_t /* no limit on majors.size(), so edge_t can overflow */,
                                   multi_gpu,
                                   true>{this->offsets_,
                                         major_range_first_,
                                         *dcs_nzd_vertices_,
                                         *major_hypersparse_first_},
                                 size_t{0},
                                 thrust::plus<size_t>())
                             : thrust::transform_reduce(
                                 rmm::exec_policy(stream),
                                 majors.begin(),
                                 majors.end(),
                                 detail::local_degree_op_t<
                                   vertex_t,
                                   edge_t,
                                   size_t /* no limit on majors.size(), so edge_t can overflow */,
                                   multi_gpu,
                                   false>{this->offsets_,
                                          major_range_first_,
                                          std::byte{0} /* dummy */,
                                          std::byte{0} /* dummy */},
                                 size_t{0},
                                 thrust::plus<size_t>());
  }

  rmm::device_uvector<edge_t> compute_local_degrees(rmm::cuda_stream_view stream) const
  {
    rmm::device_uvector<edge_t> local_degrees(this->major_range_size(), stream);
    if (dcs_nzd_vertices_) {
      assert(major_hypersparse_first_);
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator(this->major_range_first()),
                        thrust::make_counting_iterator(this->major_range_last()),
                        local_degrees.begin(),
                        detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, true>{
                          this->offsets_,
                          major_range_first_,
                          *dcs_nzd_vertices_,
                          major_hypersparse_first_.value_or(vertex_t{0})});
    } else {
      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(this->major_range_first()),
        thrust::make_counting_iterator(this->major_range_last()),
        local_degrees.begin(),
        detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false>{
          this->offsets_, major_range_first_, std::byte{0} /* dummy */, std::byte{0} /* dummy */});
    }
    return local_degrees;
  }

  rmm::device_uvector<edge_t> compute_local_degrees(raft::device_span<vertex_t const> majors,
                                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_uvector<edge_t> local_degrees(majors.size(), stream);
    if (dcs_nzd_vertices_) {
      assert(major_hypersparse_first_);
      thrust::transform(rmm::exec_policy(stream),
                        majors.begin(),
                        majors.end(),
                        local_degrees.begin(),
                        detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, true>{
                          this->offsets_,
                          major_range_first_,
                          dcs_nzd_vertices_.value(),
                          major_hypersparse_first_.value_or(vertex_t{0})});
    } else {
      thrust::transform(
        rmm::exec_policy(stream),
        majors.begin(),
        majors.end(),
        local_degrees.begin(),
        detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false>{
          this->offsets_, major_range_first_, std::byte{0} /* dummy */, std::byte{0} /* dummy */});
    }
    return local_degrees;
  }

  __host__ __device__ vertex_t major_value_start_offset() const
  {
    return major_value_start_offset_;
  }

  __host__ __device__ thrust::optional<vertex_t> major_hypersparse_first() const noexcept
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

  __host__ __device__ vertex_t minor_rage_last() const noexcept { return minor_range_last_; }

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

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> major_hypersparse_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    if (dcs_nzd_vertices_) {
      return detail::major_hypersparse_idx_from_major_nocheck_impl(*dcs_nzd_vertices_, major);
    } else {
      return thrust::nullopt;
    }
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> major_from_major_hypersparse_idx_nocheck(
    vertex_t major_hypersparse_idx) const noexcept
  {
    return dcs_nzd_vertices_
             ? thrust::optional<vertex_t>{(*dcs_nzd_vertices_)[major_hypersparse_idx]}
             : thrust::nullopt;
  }

  __host__ __device__ vertex_t minor_from_minor_offset_nocheck(vertex_t minor_offset) const noexcept
  {
    return minor_range_first_ + minor_offset;
  }

  __host__ __device__ thrust::optional<vertex_t const*> dcs_nzd_vertices() const
  {
    return dcs_nzd_vertices_ ? thrust::optional<vertex_t const*>{(*dcs_nzd_vertices_).data()}
                             : thrust::nullopt;
  }
  __host__ __device__ thrust::optional<vertex_t> dcs_nzd_vertex_count() const
  {
    return dcs_nzd_vertices_
             ? thrust::optional<vertex_t>{static_cast<vertex_t>((*dcs_nzd_vertices_).size())}
             : thrust::nullopt;
  }

 private:
  // should be trivially copyable to device

  thrust::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices_{thrust::nullopt};
  thrust::optional<vertex_t> major_hypersparse_first_{thrust::nullopt};

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

  size_t compute_number_of_edges(raft::device_span<vertex_t const> majors,
                                 rmm::cuda_stream_view stream) const
  {
    return thrust::transform_reduce(
      rmm::exec_policy(stream),
      majors.begin(),
      majors.end(),
      detail::local_degree_op_t<vertex_t,
                                edge_t,
                                size_t /* no limit on majors.size(), so edge_t can overflow */,
                                multi_gpu,
                                false>{this->offsets_,
                                       std::byte{0} /* dummy */,
                                       std::byte{0} /* dummy */,
                                       std::byte{0} /* dummy */},
      size_t{0},
      thrust::plus<size_t>());
  }

  rmm::device_uvector<edge_t> compute_local_degrees(rmm::cuda_stream_view stream) const
  {
    rmm::device_uvector<edge_t> local_degrees(this->major_range_size(), stream);
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(this->major_range_first()),
                      thrust::make_counting_iterator(this->major_range_last()),
                      local_degrees.begin(),
                      detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false>{
                        this->offsets_,
                        std::byte{0} /* dummy */,
                        std::byte{0} /* dummy */,
                        std::byte{0} /* dummy */});
    return local_degrees;
  }

  rmm::device_uvector<edge_t> compute_local_degrees(raft::device_span<vertex_t const> majors,
                                                    rmm::cuda_stream_view stream) const
  {
    rmm::device_uvector<edge_t> local_degrees(majors.size(), stream);
    thrust::transform(rmm::exec_policy(stream),
                      majors.begin(),
                      majors.end(),
                      local_degrees.begin(),
                      detail::local_degree_op_t<vertex_t, edge_t, edge_t, multi_gpu, false>{
                        this->offsets_,
                        std::byte{0} /* dummy */,
                        std::byte{0} /* dummy */,
                        std::byte{0} /* dummy */});
    return local_degrees;
  }

  __host__ __device__ vertex_t major_value_start_offset() const { return vertex_t{0}; }

  __host__ __device__ thrust::optional<vertex_t> major_hypersparse_first() const noexcept
  {
    assert(false);
    return thrust::nullopt;
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

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> major_hypersparse_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    assert(false);
    return thrust::nullopt;
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> major_from_major_hypersparse_idx_nocheck(
    vertex_t major_hypersparse_idx) const noexcept
  {
    assert(false);
    return thrust::nullopt;
  }

  __host__ __device__ vertex_t minor_from_minor_offset_nocheck(vertex_t minor_offset) const noexcept
  {
    return minor_offset;
  }

  __host__ __device__ thrust::optional<vertex_t const*> dcs_nzd_vertices() const
  {
    return thrust::nullopt;
  }

  __host__ __device__ thrust::optional<vertex_t> dcs_nzd_vertex_count() const
  {
    return thrust::nullopt;
  }

 private:
  vertex_t number_of_vertices_{};
};

}  // namespace cugraph

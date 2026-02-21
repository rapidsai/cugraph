/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/set_operations.h>
#include <thrust/tabulate.h>

#include <optional>

namespace cugraph {
namespace test {

template <typename vertex_t>
struct in_region_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  vertex_t v_offset_min_threshold;  // inclusive
  vertex_t v_offset_max_threshold;  // exclusive

  __device__ bool operator()(vertex_t v) const
  {
    auto vertex_partition_id = static_cast<int>(cuda::std::distance(
      vertex_partition_range_lasts.begin(),
      thrust::upper_bound(
        thrust::seq, vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.end(), v)));
    auto v_offset =
      v - ((vertex_partition_id == 0) ? vertex_t{0}
                                      : vertex_partition_range_lasts[vertex_partition_id - 1]);
    return (v_offset >= v_offset_min_threshold) && (v_offset < v_offset_max_threshold);
  }
};

template <typename vertex_t>
class nbr_unrenumber_cache_t {
 public:
  static size_t constexpr consecutive_total_bits = 18;

  static size_t constexpr dense_lsb_bits   = 8;
  using dense_lsb_t                        = uint8_t;
  static size_t constexpr dense_total_bits = 21;
  using dense_offset_t                     = uint32_t;

  static size_t constexpr sparse_lsb_bits = 16;
  using sparse_lsb_t                      = uint16_t;
  using sparse_offset_t = uint32_t;  // should be large enough to cover the maximum number of sorted
                                     // unique neighbors per vertex partition

  static_assert(consecutive_total_bits <= dense_total_bits);
  static_assert(dense_lsb_bits == sizeof(dense_lsb_t) * 8);
  static_assert(sizeof(dense_offset_t) * 8 >= dense_total_bits);
  static_assert(sparse_lsb_bits == sizeof(sparse_lsb_t) * 8);
  static_assert(consecutive_total_bits >= dense_lsb_bits);  // to ensure proper alignment
  static_assert(dense_total_bits >= sparse_lsb_bits);       // to ensure proper alignment

  nbr_unrenumber_cache_t(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>&& sorted_unique_nbrs,
                         rmm::device_uvector<vertex_t>&& unrenumbered_sorted_unique_nbrs,
                         rmm::device_uvector<vertex_t>&& vertex_partition_range_lasts,
                         vertex_t invalid_vertex,
                         std::optional<cugraph::large_buffer_type_t> large_buffer_type)
    : consecutive_unrenumbered_sorted_unique_nbrs_(0, handle.get_stream()),
      dense_sorted_unique_nbr_segment_lasts_(0, handle.get_stream()),
      dense_sorted_unique_nbr_v_offset_lsbs_(0, handle.get_stream()),
      dense_unrenumbered_sorted_unique_nbrs_(0, handle.get_stream()),
      dense_sorted_unique_nbr_vertex_partition_range_lasts_(0, handle.get_stream()),
      sparse_sorted_unique_nbr_segment_lasts_(0, handle.get_stream()),
      sparse_sorted_unique_nbr_v_offset_lsbs_(0, handle.get_stream()),
      sparse_unrenumbered_sorted_unique_nbrs_(0, handle.get_stream()),
      sparse_sorted_unique_nbr_vertex_partition_range_lasts_(0, handle.get_stream()),
      vertex_partition_range_lasts_(std::move(vertex_partition_range_lasts)),
      invalid_vertex_(invalid_vertex)
  {
    init_key_cache(handle,
                   std::move(sorted_unique_nbrs),
                   std::move(unrenumbered_sorted_unique_nbrs),
                   large_buffer_type);
  }

  void init_key_cache(raft::handle_t const& handle,
                      rmm::device_uvector<vertex_t>&& sorted_unique_nbrs,
                      rmm::device_uvector<vertex_t>&& unrenumbered_sorted_unique_nbrs,
                      std::optional<cugraph::large_buffer_type_t> large_buffer_type)
  {
    CUGRAPH_EXPECTS(
      !large_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
      "Invalid input argument: large memory buffer is not initialized.");

    vertex_t max_vertex_partition_size{};
    {
      auto size_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<vertex_t>(
          [lasts = raft::device_span<vertex_t const>(
             vertex_partition_range_lasts_.data(),
             vertex_partition_range_lasts_.size())] __device__(auto i) {
            return (i == 0) ? lasts[i] : (lasts[i] - lasts[i - 1]);
          }));
      max_vertex_partition_size = thrust::reduce(handle.get_thrust_policy(),
                                                 size_first,
                                                 size_first + vertex_partition_range_lasts_.size(),
                                                 vertex_t{0},
                                                 thrust::maximum<vertex_t>{});
    }

    consecutive_size_per_vertex_partition_ =
      std::min(max_vertex_partition_size, vertex_t{1} << consecutive_total_bits);
    size_t dense_sorted_unique_nbr_size{};
    size_t sparse_sorted_unique_nbr_size{};
    {
      auto in_dense_func = in_region_t<vertex_t>{
        raft::device_span<vertex_t const>(vertex_partition_range_lasts_.data(),
                                          vertex_partition_range_lasts_.size()),
        consecutive_size_per_vertex_partition_,
        vertex_t{1} << dense_total_bits};
      auto in_sparse_func = in_region_t<vertex_t>{
        raft::device_span<vertex_t const>(vertex_partition_range_lasts_.data(),
                                          vertex_partition_range_lasts_.size()),
        vertex_t{1} << dense_total_bits,
        std::numeric_limits<vertex_t>::max()};
      dense_sorted_unique_nbr_size  = thrust::count_if(handle.get_thrust_policy(),
                                                      sorted_unique_nbrs.begin(),
                                                      sorted_unique_nbrs.end(),
                                                      in_dense_func);
      sparse_sorted_unique_nbr_size = thrust::count_if(handle.get_thrust_policy(),
                                                       sorted_unique_nbrs.begin(),
                                                       sorted_unique_nbrs.end(),
                                                       in_sparse_func);

      auto pair_first = thrust::make_zip_iterator(sorted_unique_nbrs.begin(),
                                                  unrenumbered_sorted_unique_nbrs.begin());

      consecutive_unrenumbered_sorted_unique_nbrs_.resize(
        consecutive_size_per_vertex_partition_ * vertex_partition_range_lasts_.size(),
        handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   consecutive_unrenumbered_sorted_unique_nbrs_.begin(),
                   consecutive_unrenumbered_sorted_unique_nbrs_.end(),
                   invalid_vertex_);
      thrust::for_each(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + sorted_unique_nbrs.size(),
        cuda::proclaim_return_type<void>(
          [consecutive_unrenumbered_sorted_unique_nbrs =
             raft::device_span<vertex_t>(consecutive_unrenumbered_sorted_unique_nbrs_.data(),
                                         consecutive_unrenumbered_sorted_unique_nbrs_.size()),
           vertex_partition_range_lasts = raft::device_span<vertex_t const>(
             vertex_partition_range_lasts_.data(), vertex_partition_range_lasts_.size()),
           consecutive_size_per_vertex_partition =
             consecutive_size_per_vertex_partition_] __device__(auto pair) {
            auto v                   = cuda::std::get<0>(pair);
            auto vertex_partition_id = static_cast<int>(
              cuda::std::distance(vertex_partition_range_lasts.begin(),
                                  thrust::upper_bound(thrust::seq,
                                                      vertex_partition_range_lasts.begin(),
                                                      vertex_partition_range_lasts.end(),
                                                      v)));
            auto v_offset = v - ((vertex_partition_id == 0)
                                   ? vertex_t{0}
                                   : vertex_partition_range_lasts[vertex_partition_id - 1]);
            if (v_offset < consecutive_size_per_vertex_partition) {
              consecutive_unrenumbered_sorted_unique_nbrs[consecutive_size_per_vertex_partition *
                                                            vertex_partition_id +
                                                          v_offset] = cuda::std::get<1>(pair);
            }
          }));

      auto tmp_sorted_unique_nbrs =
        large_buffer_type
          ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
              dense_sorted_unique_nbr_size + sparse_sorted_unique_nbr_size, handle.get_stream())
          : rmm::device_uvector<vertex_t>(
              dense_sorted_unique_nbr_size + sparse_sorted_unique_nbr_size, handle.get_stream());
      dense_unrenumbered_sorted_unique_nbrs_.resize(dense_sorted_unique_nbr_size,
                                                    handle.get_stream());
      sparse_unrenumbered_sorted_unique_nbrs_.resize(sparse_sorted_unique_nbr_size,
                                                     handle.get_stream());
      thrust::copy_if(handle.get_thrust_policy(),
                      pair_first,
                      pair_first + sorted_unique_nbrs.size(),
                      thrust::make_zip_iterator(tmp_sorted_unique_nbrs.begin(),
                                                dense_unrenumbered_sorted_unique_nbrs_.begin()),
                      cuda::proclaim_return_type<bool>([in_dense_func] __device__(auto pair) {
                        return in_dense_func(cuda::std::get<0>(pair));
                      }));
      thrust::copy_if(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + sorted_unique_nbrs.size(),
        thrust::make_zip_iterator(tmp_sorted_unique_nbrs.begin() + dense_sorted_unique_nbr_size,
                                  sparse_unrenumbered_sorted_unique_nbrs_.begin()),
        cuda::proclaim_return_type<bool>([in_sparse_func] __device__(auto pair) {
          return in_sparse_func(cuda::std::get<0>(pair));
        }));

      sorted_unique_nbrs = std::move(tmp_sorted_unique_nbrs);
      unrenumbered_sorted_unique_nbrs.resize(0, handle.get_stream());
      unrenumbered_sorted_unique_nbrs.shrink_to_fit(handle.get_stream());
    }
    dense_sorted_unique_nbr_vertex_partition_range_lasts_.resize(
      vertex_partition_range_lasts_.size(), handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        sorted_unique_nbrs.begin(),
                        sorted_unique_nbrs.begin() + dense_sorted_unique_nbr_size,
                        vertex_partition_range_lasts_.begin(),
                        vertex_partition_range_lasts_.end(),
                        dense_sorted_unique_nbr_vertex_partition_range_lasts_.begin());

    sparse_sorted_unique_nbr_vertex_partition_range_lasts_.resize(
      vertex_partition_range_lasts_.size(), handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        sorted_unique_nbrs.begin() + dense_sorted_unique_nbr_size,
                        sorted_unique_nbrs.end(),
                        vertex_partition_range_lasts_.begin(),
                        vertex_partition_range_lasts_.end(),
                        sparse_sorted_unique_nbr_vertex_partition_range_lasts_.begin());

    auto sorted_unique_nbr_v_offsets =
      large_buffer_type
        ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(sorted_unique_nbrs.size(),
                                                                          handle.get_stream())
        : rmm::device_uvector<vertex_t>(sorted_unique_nbrs.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      sorted_unique_nbrs.begin(),
                      sorted_unique_nbrs.end(),
                      sorted_unique_nbr_v_offsets.begin(),
                      cuda::proclaim_return_type<vertex_t>(
                        [vertex_partition_range_lasts = raft::device_span<vertex_t const>(
                           vertex_partition_range_lasts_.data(),
                           vertex_partition_range_lasts_.size())] __device__(auto v) {
                          auto vertex_partition_id = static_cast<int>(cuda::std::distance(
                            vertex_partition_range_lasts.begin(),
                            thrust::upper_bound(thrust::seq,
                                                vertex_partition_range_lasts.begin(),
                                                vertex_partition_range_lasts.end(),
                                                v)));
                          return v - ((vertex_partition_id == 0)
                                        ? vertex_t{0}
                                        : vertex_partition_range_lasts[vertex_partition_id - 1]);
                        }));
    sorted_unique_nbrs.resize(0, handle.get_stream());
    sorted_unique_nbrs.shrink_to_fit(handle.get_stream());

    {
      auto dense_size_per_vertex_partition =
        std::min(vertex_t{1} << dense_total_bits, max_vertex_partition_size) -
        consecutive_size_per_vertex_partition_;
      auto sparse_size_per_vertex_partition =
        (max_vertex_partition_size >
         (dense_size_per_vertex_partition + consecutive_size_per_vertex_partition_))
          ? (max_vertex_partition_size -
             (dense_size_per_vertex_partition + consecutive_size_per_vertex_partition_))
          : vertex_t{0};
      if ((sparse_size_per_vertex_partition != 0) &&
          ((sparse_size_per_vertex_partition - 1) >
           static_cast<vertex_t>(std::numeric_limits<sparse_offset_t>::max()))) {
        CUGRAPH_FAIL("sparse_offset_t overflow.");
      }

      dense_num_segments_per_vertex_partition_ =
        (static_cast<size_t>(dense_size_per_vertex_partition) +
         ((size_t{1} << dense_lsb_bits) - 1)) >>
        dense_lsb_bits;
      sparse_num_segments_per_vertex_partition_ =
        (static_cast<size_t>(sparse_size_per_vertex_partition) +
         ((size_t{1} << sparse_lsb_bits) - 1)) >>
        sparse_lsb_bits;
    }

    dense_sorted_unique_nbr_segment_lasts_.resize(
      dense_num_segments_per_vertex_partition_ * vertex_partition_range_lasts_.size(),
      handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      dense_sorted_unique_nbr_segment_lasts_.begin(),
      dense_sorted_unique_nbr_segment_lasts_.end(),
      cuda::proclaim_return_type<dense_offset_t>(
        [sorted_unique_nbr_v_offsets = raft::device_span<vertex_t const>(
           sorted_unique_nbr_v_offsets.data(), dense_sorted_unique_nbr_size),
         sorted_unique_nbr_vertex_partition_range_lasts = raft::device_span<vertex_t const>(
           dense_sorted_unique_nbr_vertex_partition_range_lasts_.data(),
           dense_sorted_unique_nbr_vertex_partition_range_lasts_.size()),
         vertex_partition_range_lasts = raft::device_span<vertex_t const>(
           vertex_partition_range_lasts_.data(), vertex_partition_range_lasts_.size()),
         dense_num_segments_per_vertex_partition =
           dense_num_segments_per_vertex_partition_] __device__(size_t i) {
          auto vertex_partition_id = static_cast<int>(i / dense_num_segments_per_vertex_partition);
          auto segment_id          = i % dense_num_segments_per_vertex_partition;
          auto segment_v_offset_last = static_cast<vertex_t>(cuda::std::min(
            (size_t{1} << consecutive_total_bits) + ((segment_id + 1) << dense_lsb_bits),
            static_cast<size_t>(
              vertex_partition_range_lasts[vertex_partition_id] -
              ((vertex_partition_id == 0)
                 ? vertex_t{0}
                 : vertex_partition_range_lasts
                     [vertex_partition_id - 1]))));  // consecutive_total_bits >= dense_lsb_bits, so
                                                     // proper alignment is guaranteed
          auto start_offset =
            ((vertex_partition_id == 0)
               ? vertex_t{0}
               : sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id - 1]);
          auto end_offset = sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id];
          auto it         = thrust::lower_bound(thrust::seq,
                                        sorted_unique_nbr_v_offsets.begin() + start_offset,
                                        sorted_unique_nbr_v_offsets.begin() + end_offset,
                                        segment_v_offset_last);
          return static_cast<dense_offset_t>(
            cuda::std::distance(sorted_unique_nbr_v_offsets.begin() + start_offset, it));
        }));

    dense_sorted_unique_nbr_v_offset_lsbs_.resize(dense_sorted_unique_nbr_size,
                                                  handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      sorted_unique_nbr_v_offsets.begin(),
                      sorted_unique_nbr_v_offsets.begin() + dense_sorted_unique_nbr_size,
                      dense_sorted_unique_nbr_v_offset_lsbs_.begin(),
                      cuda::proclaim_return_type<dense_lsb_t>([] __device__(auto v_offset) {
                        return static_cast<dense_lsb_t>(static_cast<uint64_t>(v_offset));
                      }));

    sparse_sorted_unique_nbr_segment_lasts_.resize(
      sparse_num_segments_per_vertex_partition_ * vertex_partition_range_lasts_.size(),
      handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      sparse_sorted_unique_nbr_segment_lasts_.begin(),
      sparse_sorted_unique_nbr_segment_lasts_.end(),
      cuda::proclaim_return_type<sparse_offset_t>(
        [sorted_unique_nbr_v_offsets = raft::device_span<vertex_t const>(
           sorted_unique_nbr_v_offsets.data() + dense_sorted_unique_nbr_size,
           sparse_sorted_unique_nbr_size),
         sorted_unique_nbr_vertex_partition_range_lasts = raft::device_span<vertex_t const>(
           sparse_sorted_unique_nbr_vertex_partition_range_lasts_.data(),
           sparse_sorted_unique_nbr_vertex_partition_range_lasts_.size()),
         vertex_partition_range_lasts = raft::device_span<vertex_t const>(
           vertex_partition_range_lasts_.data(), vertex_partition_range_lasts_.size()),
         sparse_num_segments_per_vertex_partition =
           sparse_num_segments_per_vertex_partition_] __device__(size_t i) {
          auto vertex_partition_id = static_cast<int>(i / sparse_num_segments_per_vertex_partition);
          auto segment_id          = i % sparse_num_segments_per_vertex_partition;
          auto segment_v_offset_last = static_cast<vertex_t>(cuda::std::min(
            (size_t{1} << dense_total_bits) + ((segment_id + 1) << sparse_lsb_bits),
            static_cast<size_t>(
              vertex_partition_range_lasts[vertex_partition_id] -
              ((vertex_partition_id == 0)
                 ? vertex_t{0}
                 : vertex_partition_range_lasts[vertex_partition_id -
                                                1]))));  // dense_total_bits >= sparse_lsb_bits, so
                                                         // proper alignment is guaranteed
          auto start_offset =
            ((vertex_partition_id == 0)
               ? vertex_t{0}
               : sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id - 1]);
          auto end_offset = sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id];
          auto it         = thrust::lower_bound(thrust::seq,
                                        sorted_unique_nbr_v_offsets.begin() + start_offset,
                                        sorted_unique_nbr_v_offsets.begin() + end_offset,
                                        segment_v_offset_last);
          return static_cast<sparse_offset_t>(
            cuda::std::distance(sorted_unique_nbr_v_offsets.begin() + start_offset, it));
        }));

    sparse_sorted_unique_nbr_v_offset_lsbs_.resize(sparse_sorted_unique_nbr_size,
                                                   handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      sorted_unique_nbr_v_offsets.begin() + dense_sorted_unique_nbr_size,
                      sorted_unique_nbr_v_offsets.end(),
                      sparse_sorted_unique_nbr_v_offset_lsbs_.begin(),
                      cuda::proclaim_return_type<sparse_lsb_t>([] __device__(auto v_offset) {
                        return static_cast<sparse_lsb_t>(static_cast<uint64_t>(v_offset));
                      }));

    sorted_unique_nbr_v_offsets.resize(0, handle.get_stream());
    sorted_unique_nbr_v_offsets.shrink_to_fit(handle.get_stream());
  }

  void unrenumber(raft::handle_t const& handle, raft::device_span<vertex_t> nbrs)
  {
    thrust::transform_if(
      handle.get_thrust_policy(),
      nbrs.begin(),
      nbrs.end(),
      nbrs.begin(),
      cuda::proclaim_return_type<
        vertex_t>([consecutive_unrenumbered_sorted_unique_nbrs = raft::device_span<vertex_t const>(
                     consecutive_unrenumbered_sorted_unique_nbrs_.data(),
                     consecutive_unrenumbered_sorted_unique_nbrs_.size()),
                   consecutive_size_per_vertex_partition = consecutive_size_per_vertex_partition_,
                   dense_sorted_unique_nbr_segment_lasts = raft::device_span<dense_offset_t const>(
                     dense_sorted_unique_nbr_segment_lasts_.data(),
                     dense_sorted_unique_nbr_segment_lasts_.size()),
                   dense_sorted_unique_nbr_v_offset_lsbs = raft::device_span<dense_lsb_t const>(
                     dense_sorted_unique_nbr_v_offset_lsbs_.data(),
                     dense_sorted_unique_nbr_v_offset_lsbs_.size()),
                   dense_unrenumbered_sorted_unique_nbrs = raft::device_span<vertex_t const>(
                     dense_unrenumbered_sorted_unique_nbrs_.data(),
                     dense_unrenumbered_sorted_unique_nbrs_.size()),
                   dense_sorted_unique_nbr_vertex_partition_range_lasts =
                     raft::device_span<vertex_t const>(
                       dense_sorted_unique_nbr_vertex_partition_range_lasts_.data(),
                       dense_sorted_unique_nbr_vertex_partition_range_lasts_.size()),
                   dense_num_segments_per_vertex_partition =
                     dense_num_segments_per_vertex_partition_,
                   sparse_sorted_unique_nbr_segment_lasts =
                     raft::device_span<sparse_offset_t const>(
                       sparse_sorted_unique_nbr_segment_lasts_.data(),
                       sparse_sorted_unique_nbr_segment_lasts_.size()),
                   sparse_sorted_unique_nbr_v_offset_lsbs = raft::device_span<sparse_lsb_t const>(
                     sparse_sorted_unique_nbr_v_offset_lsbs_.data(),
                     sparse_sorted_unique_nbr_v_offset_lsbs_.size()),
                   sparse_unrenumbered_sorted_unique_nbrs = raft::device_span<vertex_t const>(
                     sparse_unrenumbered_sorted_unique_nbrs_.data(),
                     sparse_unrenumbered_sorted_unique_nbrs_.size()),
                   sparse_sorted_unique_nbr_vertex_partition_range_lasts =
                     raft::device_span<vertex_t const>(
                       sparse_sorted_unique_nbr_vertex_partition_range_lasts_.data(),
                       sparse_sorted_unique_nbr_vertex_partition_range_lasts_.size()),
                   sparse_num_segments_per_vertex_partition =
                     sparse_num_segments_per_vertex_partition_,
                   vertex_partition_range_lasts = raft::device_span<vertex_t const>(
                     vertex_partition_range_lasts_.data(),
                     vertex_partition_range_lasts_.size())] __device__(auto nbr) {
        auto vertex_partition_id = static_cast<int>(
          cuda::std::distance(vertex_partition_range_lasts.begin(),
                              thrust::upper_bound(thrust::seq,
                                                  vertex_partition_range_lasts.begin(),
                                                  vertex_partition_range_lasts.end(),
                                                  nbr)));
        auto v_offset = nbr - ((vertex_partition_id == 0)
                                 ? vertex_t{0}
                                 : vertex_partition_range_lasts[vertex_partition_id - 1]);
        if (v_offset < (vertex_t{1} << consecutive_total_bits)) {
          return consecutive_unrenumbered_sorted_unique_nbrs[consecutive_size_per_vertex_partition *
                                                               vertex_partition_id +
                                                             v_offset];
        } else {
          bool dense = v_offset < (vertex_t{1} << dense_total_bits);
          auto segment_id =
            static_cast<size_t>(
              v_offset - (vertex_t{1} << (dense ? consecutive_total_bits : dense_total_bits))) >>
            (dense ? dense_lsb_bits : sparse_lsb_bits);
          auto start_offset =
            ((vertex_partition_id == 0)
               ? vertex_t{0}
               : (dense
                    ? dense_sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id - 1]
                    : sparse_sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id -
                                                                            1])) +
            ((segment_id == 0) ? vertex_t{0}
                               : (dense ? dense_sorted_unique_nbr_segment_lasts
                                            [dense_num_segments_per_vertex_partition *
                                               static_cast<size_t>(vertex_partition_id) +
                                             segment_id - 1]
                                        : sparse_sorted_unique_nbr_segment_lasts
                                            [sparse_num_segments_per_vertex_partition *
                                               static_cast<size_t>(vertex_partition_id) +
                                             segment_id - 1]));
          auto end_offset =
            ((vertex_partition_id == 0)
               ? vertex_t{0}
               : (dense
                    ? dense_sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id - 1]
                    : sparse_sorted_unique_nbr_vertex_partition_range_lasts[vertex_partition_id -
                                                                            1])) +
            (dense
               ? dense_sorted_unique_nbr_segment_lasts[dense_num_segments_per_vertex_partition *
                                                         static_cast<size_t>(vertex_partition_id) +
                                                       segment_id]
               : sparse_sorted_unique_nbr_segment_lasts[sparse_num_segments_per_vertex_partition *
                                                          static_cast<size_t>(vertex_partition_id) +
                                                        segment_id]);
          if (dense) {
            auto it =
              thrust::lower_bound(thrust::seq,
                                  dense_sorted_unique_nbr_v_offset_lsbs.begin() + start_offset,
                                  dense_sorted_unique_nbr_v_offset_lsbs.begin() + end_offset,
                                  static_cast<dense_lsb_t>(static_cast<uint64_t>(v_offset)));
            return dense_unrenumbered_sorted_unique_nbrs[cuda::std::distance(
              dense_sorted_unique_nbr_v_offset_lsbs.begin(), it)];
          } else {
            auto it =
              thrust::lower_bound(thrust::seq,
                                  sparse_sorted_unique_nbr_v_offset_lsbs.begin() + start_offset,
                                  sparse_sorted_unique_nbr_v_offset_lsbs.begin() + end_offset,
                                  static_cast<sparse_lsb_t>(static_cast<uint64_t>(v_offset)));
            return sparse_unrenumbered_sorted_unique_nbrs[cuda::std::distance(
              sparse_sorted_unique_nbr_v_offset_lsbs.begin(), it)];
          }
        }
      }),
      cuda::proclaim_return_type<bool>(
        [invalid_vertex = invalid_vertex_] __device__(auto nbr) { return nbr != invalid_vertex; }));
  }

 private:
  rmm::device_uvector<vertex_t> consecutive_unrenumbered_sorted_unique_nbrs_;
  vertex_t consecutive_size_per_vertex_partition_{};

  rmm::device_uvector<dense_offset_t> dense_sorted_unique_nbr_segment_lasts_;
  rmm::device_uvector<dense_lsb_t> dense_sorted_unique_nbr_v_offset_lsbs_;
  rmm::device_uvector<vertex_t> dense_unrenumbered_sorted_unique_nbrs_;
  rmm::device_uvector<vertex_t> dense_sorted_unique_nbr_vertex_partition_range_lasts_;
  size_t dense_num_segments_per_vertex_partition_{};

  rmm::device_uvector<sparse_offset_t> sparse_sorted_unique_nbr_segment_lasts_;
  rmm::device_uvector<sparse_lsb_t> sparse_sorted_unique_nbr_v_offset_lsbs_;
  rmm::device_uvector<vertex_t> sparse_unrenumbered_sorted_unique_nbrs_;
  rmm::device_uvector<vertex_t> sparse_sorted_unique_nbr_vertex_partition_range_lasts_;
  size_t sparse_num_segments_per_vertex_partition_{};

  rmm::device_uvector<vertex_t> vertex_partition_range_lasts_;
  vertex_t invalid_vertex_{};
};

template <typename vertex_t, typename edge_t>
nbr_unrenumber_cache_t<vertex_t> build_nbr_unrenumber_cache(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_graph_view,
  raft::device_span<vertex_t const> renumber_map,
  vertex_t invalid_vertex,
  std::optional<cugraph::large_buffer_type_t> large_buffer_type)
{
  constexpr size_t num_k_hop_rounds      = 24;  // to cut peak memory usage
  constexpr size_t num_unrenumber_rounds = 8;   // to cut peak memory usage

  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  CUGRAPH_EXPECTS(!large_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
                  "Large memory buffer is not initialized.");
  CUGRAPH_EXPECTS(
    minor_comm_size <= std::numeric_limits<uint16_t>::max(),
    "minor_comm_size is too large for uint16_t, use int instead.");  // we use uint16_t to
                                                                     // temporarily store minor_comm
                                                                     // ranks in this code
  std::optional<vertex_t> dense_size_per_vertex_partition{std::nullopt};
  auto segment_offsets = mg_graph_view.local_vertex_partition_segment_offsets();
  if (segment_offsets) {
    dense_size_per_vertex_partition = (*segment_offsets)[2] /* high & mid segments */;
#if 1  // FIXME: we should add host_allreduce to raft
    dense_size_per_vertex_partition = host_scalar_allreduce(
      comm, *dense_size_per_vertex_partition, raft::comms::op_t::SUM, handle.get_stream());
#else
    comm.host_allreduce(std::addressof(*dense_size_per_vertex_partition),
                        std::addressof(*dense_size_per_vertex_partition),
                        size_t{1},
                        raft::comms::op_t::SUM);
#endif
    *dense_size_per_vertex_partition /= comm_size;
  }

  auto h_vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();
  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
    h_vertex_partition_range_lasts.size(), handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      h_vertex_partition_range_lasts.data(),
                      h_vertex_partition_range_lasts.size(),
                      handle.get_stream());

  auto nbrs = large_buffer_type ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
                                    0, handle.get_stream())
                                : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  std::optional<rmm::device_uvector<uint32_t>> dense_nbr_bitmap{std::nullopt};
  if (dense_size_per_vertex_partition) {
    dense_nbr_bitmap = rmm::device_uvector<uint32_t>(
      cugraph::packed_bool_size(*dense_size_per_vertex_partition * comm_size), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 dense_nbr_bitmap->begin(),
                 dense_nbr_bitmap->end(),
                 cugraph::packed_bool_empty_mask());
  }
  for (size_t r = 0; r < num_k_hop_rounds; ++r) {
    rmm::device_uvector<vertex_t> this_round_nbrs(0, handle.get_stream());
    rmm::device_uvector<size_t> d_tx_counts(0, handle.get_stream());
    {
      auto range_size = mg_graph_view.local_vertex_partition_range_size();
      auto num_seeds =
        range_size / num_k_hop_rounds + (r < (range_size % num_k_hop_rounds) ? 1 : 0);
      rmm::device_uvector<vertex_t> seeds(num_seeds, handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        seeds.begin(),
        seeds.end(),
        cuda::proclaim_return_type<vertex_t>(
          [r, v_first = mg_graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
            return v_first + static_cast<vertex_t>(r + i * num_k_hop_rounds);
          }));

      auto num_aggregate_seeds = seeds.size();
#if 1  // FIXME: we should add host_allreduce to raft
      num_aggregate_seeds = host_scalar_allreduce(
        comm, num_aggregate_seeds, raft::comms::op_t::SUM, handle.get_stream());
#else
      comm.host_allreduce(std::addressof(num_aggregate_seeds),
                          std::addressof(num_aggregate_seeds),
                          size_t{1},
                          raft::comms::op_t::SUM);
#endif
      if (num_aggregate_seeds == size_t{0}) { continue; }

      rmm::device_uvector<uint16_t> minor_comm_ranks(0, handle.get_stream());
      cugraph::key_bucket_t<vertex_t, void, true, true> frontier(handle, std::move(seeds));
      std::tie(minor_comm_ranks, this_round_nbrs) =
        cugraph::extract_transform_v_frontier_outgoing_e(
          handle,
          mg_graph_view,
          frontier,
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<cuda::std::tuple<uint16_t, vertex_t>>(
            [lasts = raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                       d_vertex_partition_range_lasts.size()),
             major_comm_size,
             minor_comm_size] __device__(auto src, auto dst, auto, auto, auto) {
              auto vertex_partition_id = static_cast<int>(cuda::std::distance(
                lasts.begin(), thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), src)));
              auto comm_rank =
                cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
                  major_comm_size, minor_comm_size, vertex_partition_id);
              auto minor_comm_rank =
                partition_manager::compute_minor_comm_rank_from_global_comm_rank(
                  major_comm_size, minor_comm_size, comm_rank);
              return cuda::std::make_tuple(static_cast<uint16_t>(minor_comm_rank), dst);
            }));
      d_tx_counts = cugraph::groupby_and_count(minor_comm_ranks.begin(),
                                               minor_comm_ranks.end(),
                                               this_round_nbrs.begin(),
                                               cugraph::detail::typecast_t<uint16_t, int>{},
                                               minor_comm_size,
                                               std::numeric_limits<size_t>::max(),
                                               handle.get_stream());
    }
    std::tie(this_round_nbrs, std::ignore) =
      shuffle_values(minor_comm,
                     this_round_nbrs.begin(),
                     raft::device_span<size_t const>(d_tx_counts.data(), d_tx_counts.size()),
                     handle.get_stream());
    if (dense_size_per_vertex_partition) {
      auto sparse_last = thrust::partition(
        handle.get_thrust_policy(),
        this_round_nbrs.begin(),
        this_round_nbrs.end(),
        cuda::proclaim_return_type<bool>(
          [lasts = raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                     d_vertex_partition_range_lasts.size()),
           dense_size_per_vertex_partition = *dense_size_per_vertex_partition] __device__(auto v) {
            auto vertex_partition_id = static_cast<int>(cuda::std::distance(
              lasts.begin(), thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), v)));
            auto v_first =
              (vertex_partition_id == 0) ? vertex_t{0} : lasts[vertex_partition_id - 1];
            return (v - v_first) >= dense_size_per_vertex_partition;
          }));
      thrust::for_each(
        handle.get_thrust_policy(),
        sparse_last,
        this_round_nbrs.end(),
        [lasts  = raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                   d_vertex_partition_range_lasts.size()),
         bitmap = raft::device_span<uint32_t>(dense_nbr_bitmap->data(), dense_nbr_bitmap->size()),
         dense_size_per_vertex_partition = *dense_size_per_vertex_partition] __device__(auto v) {
          auto vertex_partition_id = static_cast<int>(cuda::std::distance(
            lasts.begin(), thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), v)));
          auto v_first  = (vertex_partition_id == 0) ? vertex_t{0} : lasts[vertex_partition_id - 1];
          auto v_offset = v - v_first;
          auto offset   = dense_size_per_vertex_partition * vertex_partition_id + v_offset;
          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
            bitmap[cugraph::packed_bool_offset(offset)]);
          word.fetch_or(cugraph::packed_bool_mask(offset), cuda::std::memory_order_relaxed);
        });
      this_round_nbrs.resize(cuda::std::distance(this_round_nbrs.begin(), sparse_last),
                             handle.get_stream());
      this_round_nbrs.shrink_to_fit(handle.get_stream());
    }
    thrust::sort(handle.get_thrust_policy(), this_round_nbrs.begin(), this_round_nbrs.end());
    this_round_nbrs.resize(
      cuda::std::distance(
        this_round_nbrs.begin(),
        thrust::unique(handle.get_thrust_policy(), this_round_nbrs.begin(), this_round_nbrs.end())),
      handle.get_stream());
    if ((nbrs.size() > 0) && (this_round_nbrs.size() > 0)) {
      auto tmp_buf_size = std::max(nbrs.size() / num_k_hop_rounds, this_round_nbrs.size());
      auto num_chunks   = (nbrs.size() + (tmp_buf_size - 1)) / tmp_buf_size;
      vertex_t this_round_nbr_offset{0};
      for (size_t i = 0; i < num_chunks; ++i) {
        rmm::device_uvector<vertex_t> tmp_nbrs(
          (i == num_chunks - 1) ? (nbrs.size() - (tmp_buf_size * i)) : tmp_buf_size,
          handle.get_stream());
        raft::update_device(
          tmp_nbrs.data(), nbrs.data() + tmp_buf_size * i, tmp_nbrs.size(), handle.get_stream());
        auto last = (i == num_chunks - 1)
                      ? this_round_nbrs.end()
                      : thrust::upper_bound(handle.get_thrust_policy(),
                                            this_round_nbrs.begin() + this_round_nbr_offset,
                                            this_round_nbrs.end(),
                                            tmp_nbrs.back_element(handle.get_stream()));
        rmm::device_uvector<vertex_t> new_nbrs(
          cuda::std::distance(this_round_nbrs.begin() + this_round_nbr_offset, last),
          handle.get_stream());
        new_nbrs.resize(cuda::std::distance(
                          new_nbrs.begin(),
                          thrust::set_difference(handle.get_thrust_policy(),
                                                 this_round_nbrs.begin() + this_round_nbr_offset,
                                                 last,
                                                 tmp_nbrs.begin(),
                                                 tmp_nbrs.end(),
                                                 new_nbrs.begin())),
                        handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     new_nbrs.begin(),
                     new_nbrs.end(),
                     this_round_nbrs.begin() + this_round_nbr_offset);
        thrust::fill(handle.get_thrust_policy(),
                     this_round_nbrs.begin() + this_round_nbr_offset + new_nbrs.size(),
                     last,
                     invalid_vertex);
        this_round_nbr_offset += static_cast<vertex_t>(
          cuda::std::distance(this_round_nbrs.begin() + this_round_nbr_offset, last));
      }
      this_round_nbrs.resize(cuda::std::distance(this_round_nbrs.begin(),
                                                 thrust::remove(handle.get_thrust_policy(),
                                                                this_round_nbrs.begin(),
                                                                this_round_nbrs.end(),
                                                                invalid_vertex)),
                             handle.get_stream());
    }
    this_round_nbrs.shrink_to_fit(handle.get_stream());
    auto merged_nbrs =
      large_buffer_type
        ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
            nbrs.size() + this_round_nbrs.size(), handle.get_stream())
        : rmm::device_uvector<vertex_t>(nbrs.size() + this_round_nbrs.size(), handle.get_stream());
    thrust::merge(handle.get_thrust_policy(),
                  nbrs.begin(),
                  nbrs.end(),
                  this_round_nbrs.begin(),
                  this_round_nbrs.end(),
                  merged_nbrs.begin());
    nbrs = std::move(merged_nbrs);
  }
  if (dense_size_per_vertex_partition) {
    auto num_dense_nbrs =
      thrust::transform_reduce(handle.get_thrust_policy(),
                               dense_nbr_bitmap->begin(),
                               dense_nbr_bitmap->end(),
                               cuda::proclaim_return_type<vertex_t>([] __device__(uint32_t word) {
                                 return static_cast<vertex_t>(__popc(word));
                               }),
                               vertex_t{0},
                               thrust::plus<vertex_t>{});
    std::cout << "num_dense_nbrs=" << num_dense_nbrs << " (sparse) nbrs.size()=" << nbrs.size()
              << std::endl;
    auto dense_nbrs    = large_buffer_type
                           ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
                            num_dense_nbrs, handle.get_stream())
                           : rmm::device_uvector<vertex_t>(num_dense_nbrs, handle.get_stream());
    auto dense_v_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [lasts = raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                   d_vertex_partition_range_lasts.size()),
         dense_size_per_vertex_partition = *dense_size_per_vertex_partition] __device__(auto i) {
          auto vertex_partition_id = i / dense_size_per_vertex_partition;
          auto v_first = (vertex_partition_id == 0) ? vertex_t{0} : lasts[vertex_partition_id - 1];
          return v_first + (i % dense_size_per_vertex_partition);
        }));
    auto stencil_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      cuda::proclaim_return_type<bool>(
        [bitmap = raft::device_span<uint32_t const>(dense_nbr_bitmap->data(),
                                                    dense_nbr_bitmap->size())] __device__(auto i) {
          auto word = bitmap[cugraph::packed_bool_offset(i)];
          return (word & cugraph::packed_bool_mask(i)) != cugraph::packed_bool_empty_mask();
        }));
    thrust::copy_if(handle.get_thrust_policy(),
                    dense_v_first,
                    dense_v_first + (*dense_size_per_vertex_partition * comm_size),
                    stencil_first,
                    dense_nbrs.begin(),
                    cuda::std::identity{});
    auto merged_nbrs =
      large_buffer_type
        ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
            nbrs.size() + dense_nbrs.size(), handle.get_stream())
        : rmm::device_uvector<vertex_t>(nbrs.size() + dense_nbrs.size(), handle.get_stream());
    thrust::merge(handle.get_thrust_policy(),
                  nbrs.begin(),
                  nbrs.end(),
                  dense_nbrs.begin(),
                  dense_nbrs.end(),
                  merged_nbrs.begin());
    nbrs                            = std::move(merged_nbrs);
    dense_size_per_vertex_partition = std::nullopt;
    dense_nbr_bitmap                = std::nullopt;
  }

  auto unrenumbered_nbrs = large_buffer_type
                             ? cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
                                 nbrs.size(), handle.get_stream())
                             : rmm::device_uvector<vertex_t>(nbrs.size(), handle.get_stream());
  for (size_t r = 0; r < num_unrenumber_rounds; ++r) {
    auto chunk_size =
      (unrenumbered_nbrs.size() + (num_unrenumber_rounds - 1)) / num_unrenumber_rounds;
    rmm::device_uvector<vertex_t> this_chunk_nbrs(
      unrenumbered_nbrs.size() / num_unrenumber_rounds +
        ((r < (unrenumbered_nbrs.size() % num_unrenumber_rounds)) ? size_t{1} : size_t{0}),
      handle.get_stream());
    auto offset_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<size_t>(
        [num_unrenumber_rounds, r] __device__(auto i) { return r + i * num_unrenumber_rounds; }));
    thrust::gather(
      handle.get_thrust_policy(),
      offset_first,
      offset_first + this_chunk_nbrs.size(),
      nbrs.begin(),
      this_chunk_nbrs
        .begin());  // gather with strides to avoid most vertices falling to the local vertex
                    // partition ranges of a small subset of GPUs leading to uneven memory pressures
    auto this_chunk_unrenumbered_nbrs = cugraph::collect_values_for_sorted_unique_int_vertices(
      handle,
      raft::device_span<vertex_t const>(this_chunk_nbrs.data(), this_chunk_nbrs.size()),
      renumber_map.begin(),
      mg_graph_view.vertex_partition_range_lasts(),
      mg_graph_view.local_vertex_partition_range_first());
    this_chunk_nbrs.resize(0, handle.get_stream());
    this_chunk_nbrs.shrink_to_fit(handle.get_stream());
    thrust::scatter(handle.get_thrust_policy(),
                    this_chunk_unrenumbered_nbrs.begin(),
                    this_chunk_unrenumbered_nbrs.end(),
                    offset_first,
                    unrenumbered_nbrs.begin());
  }

  return nbr_unrenumber_cache_t(handle,
                                std::move(nbrs),
                                std::move(unrenumbered_nbrs),
                                std::move(d_vertex_partition_range_lasts),
                                invalid_vertex,
                                large_buffer_type);
}

}  // namespace test
}  // namespace cugraph

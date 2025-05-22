/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "prims/detail/multi_stream_utils.cuh"

#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cinttypes>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace cugraph {

template <typename vertex_t, typename KeyIterator>
KeyIterator compute_key_lower_bound(KeyIterator sorted_unique_key_first,
                                    KeyIterator sorted_unique_key_last,
                                    vertex_t v_threshold,
                                    rmm::cuda_stream_view stream_view)
{
  using key_t = typename thrust::iterator_traits<KeyIterator>::value_type;

  if constexpr (std::is_same_v<key_t, vertex_t>) {
    return thrust::lower_bound(
      rmm::exec_policy(stream_view), sorted_unique_key_first, sorted_unique_key_last, v_threshold);
  } else {
    key_t k_threshold{};
    thrust::get<0>(k_threshold) = v_threshold;
    return thrust::lower_bound(
      rmm::exec_policy(stream_view),
      sorted_unique_key_first,
      sorted_unique_key_last,
      k_threshold,
      [] __device__(auto lhs, auto rhs) { return thrust::get<0>(lhs) < thrust::get<0>(rhs); });
  }
}

template <typename vertex_t, typename KeyIterator>
std::vector<size_t> compute_key_segment_offsets(KeyIterator sorted_key_first,
                                                KeyIterator sorted_key_last,
                                                raft::host_span<vertex_t const> segment_offsets,
                                                vertex_t vertex_range_first,
                                                rmm::cuda_stream_view stream_view)
{
  using key_t = typename thrust::iterator_traits<KeyIterator>::value_type;

  std::vector<vertex_t> h_thresholds(segment_offsets.size() - 2);
  for (size_t i = 0; i < h_thresholds.size(); ++i) {
    h_thresholds[i] = vertex_range_first + segment_offsets[i + 1];
  }

  rmm::device_uvector<vertex_t> d_thresholds(h_thresholds.size(), stream_view);
  raft::update_device(d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), stream_view);

  rmm::device_uvector<size_t> d_offsets(d_thresholds.size(), stream_view);
  if constexpr (std::is_same_v<key_t, vertex_t>) {
    thrust::lower_bound(rmm::exec_policy_nosync(stream_view),
                        sorted_key_first,
                        sorted_key_last,
                        d_thresholds.begin(),
                        d_thresholds.end(),
                        d_offsets.begin());
  } else {
    auto sorted_vertex_first =
      thrust::make_transform_iterator(sorted_key_first, thrust_tuple_get<key_t, 0>{});
    thrust::lower_bound(
      rmm::exec_policy_nosync(stream_view),
      sorted_vertex_first,
      sorted_vertex_first + cuda::std::distance(sorted_key_first, sorted_key_last),
      d_thresholds.begin(),
      d_thresholds.end(),
      d_offsets.begin());
  }

  std::vector<size_t> h_offsets(d_offsets.size() + 2);
  raft::update_host(h_offsets.data() + 1, d_offsets.data(), d_offsets.size(), stream_view);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view));
  h_offsets[0]     = size_t{0};
  h_offsets.back() = static_cast<size_t>(cuda::std::distance(sorted_key_first, sorted_key_last));

  return h_offsets;
}

template <typename VertexIterator>
rmm::device_uvector<uint32_t> compute_vertex_list_bitmap_info(
  VertexIterator sorted_unique_vertex_first,
  VertexIterator sorted_unique_vertex_last,
  typename thrust::iterator_traits<VertexIterator>::value_type vertex_range_first,
  typename thrust::iterator_traits<VertexIterator>::value_type vertex_range_last,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;

  auto bitmap = rmm::device_uvector<uint32_t>(
    packed_bool_size(vertex_range_last - vertex_range_first), stream_view);
  rmm::device_uvector<vertex_t> lasts(bitmap.size(), stream_view);
  auto bdry_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(vertex_t{1}),
    cuda::proclaim_return_type<vertex_t>(
      [vertex_range_first,
       vertex_range_size = vertex_range_last - vertex_range_first] __device__(vertex_t i) {
        return vertex_range_first +
               static_cast<vertex_t>(
                 std::min(packed_bools_per_word() * i, static_cast<size_t>(vertex_range_size)));
      }));
  thrust::lower_bound(rmm::exec_policy_nosync(stream_view),
                      sorted_unique_vertex_first,
                      sorted_unique_vertex_last,
                      bdry_first,
                      bdry_first + bitmap.size(),
                      lasts.begin());
  thrust::tabulate(
    rmm::exec_policy_nosync(stream_view),
    bitmap.begin(),
    bitmap.end(),
    cuda::proclaim_return_type<uint32_t>(
      [sorted_unique_vertex_first,
       vertex_range_first,
       lasts = raft::device_span<vertex_t const>(lasts.data(), lasts.size())] __device__(size_t i) {
        auto offset_first = (i != 0) ? lasts[i - 1] : vertex_t{0};
        auto offset_last  = lasts[i];
        auto ret          = packed_bool_empty_mask();
        for (auto j = offset_first; j < offset_last; ++j) {
          auto v_offset = *(sorted_unique_vertex_first + j) - vertex_range_first;
          ret |= packed_bool_mask(v_offset);
        }
        return ret;
      }));

  return bitmap;
}

template <typename InputVertexIterator, typename OutputVertexIterator>
void device_bcast_vertex_list(
  raft::comms::comms_t const& comm,
  std::variant<raft::device_span<uint32_t const>, InputVertexIterator> v_list,
  OutputVertexIterator output_v_first,
  typename thrust::iterator_traits<InputVertexIterator>::value_type vertex_range_first,
  typename thrust::iterator_traits<InputVertexIterator>::value_type vertex_range_last,
  size_t v_list_size,
  int root,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<InputVertexIterator>::value_type;

  static_assert(
    std::is_same_v<typename thrust::iterator_traits<OutputVertexIterator>::value_type, vertex_t>);

  if (v_list.index() == 0) {  // bitmap
    rmm::device_uvector<uint32_t> tmp_bitmap(
      packed_bool_size(vertex_range_last - vertex_range_first), stream_view);
    assert((comm.get_rank() != root) || (std::get<0>(v_list).size() == tmp_bitmap.size()));
    device_bcast(
      comm, std::get<0>(v_list).data(), tmp_bitmap.data(), tmp_bitmap.size(), root, stream_view);
    rmm::device_scalar<size_t> dummy(size_t{0}, stream_view);  // we already know the count
    detail::copy_if_nosync(
      thrust::make_counting_iterator(vertex_range_first),
      thrust::make_counting_iterator(vertex_range_last),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(vertex_t{0}),
        cuda::proclaim_return_type<bool>(
          [bitmap = raft::device_span<uint32_t const>(
             tmp_bitmap.data(), tmp_bitmap.size())] __device__(vertex_t v_offset) {
            return ((bitmap[packed_bool_offset(v_offset)] & packed_bool_mask(v_offset)) !=
                    packed_bool_empty_mask());
          })),
      output_v_first,
      raft::device_span<size_t>(dummy.data(), size_t{1}),
      stream_view);
  } else {
    device_bcast(comm, std::get<1>(v_list), output_v_first, v_list_size, root, stream_view);
  }
}

template <typename OutputVertexIterator>
void retrieve_vertex_list_from_bitmap(
  raft::device_span<uint32_t const> bitmap,
  OutputVertexIterator output_v_first,
  raft::device_span<size_t> count /* size = 1 */,
  typename thrust::iterator_traits<OutputVertexIterator>::value_type vertex_range_first,
  typename thrust::iterator_traits<OutputVertexIterator>::value_type vertex_range_last,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<OutputVertexIterator>::value_type;

  assert((bitmap.size() >= packed_bool_size(vertex_range_last - vertex_range_first)));
  detail::copy_if_nosync(thrust::make_counting_iterator(vertex_range_first),
                         thrust::make_counting_iterator(vertex_range_last),
                         thrust::make_transform_iterator(
                           thrust::make_counting_iterator(vertex_t{0}),
                           cuda::proclaim_return_type<bool>([bitmap] __device__(vertex_t v_offset) {
                             return ((bitmap[packed_bool_offset(v_offset)] &
                                      packed_bool_mask(v_offset)) != packed_bool_empty_mask());
                           })),
                         output_v_first,
                         count,
                         stream_view);
}

// key type is either vertex_t (tag_t == void) or thrust::tuple<vertex_t, tag_t> (tag_t != void)
// if sorted_unique is true, stores unique key objects in the sorted (non-descending) order.
// if false, there can be duplicates and the elements may not be sorted.
template <typename vertex_t,
          typename tag_t     = void,
          bool multi_gpu     = false,
          bool sorted_unique = false>
class key_bucket_t {
 public:
  using key_type =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  static bool constexpr is_sorted_unique = sorted_unique;

  static_assert(std::is_same_v<tag_t, void> || std::is_arithmetic_v<tag_t>);

  using optional_variant_type =
    std::conditional_t<std::is_same_v<tag_t, void>,
                       std::byte /* dummy */,
                       std::variant<raft::device_span<tag_t const>, rmm::device_uvector<tag_t>>>;

  key_bucket_t() = delete;

  key_bucket_t(key_bucket_t const& other) = delete;

  // The default move constructor should work fine but to silence a compiler warning (CUDA 12.8)
  key_bucket_t(key_bucket_t&& other)
  {
    this->handle_ptr_ = other.handle_ptr_;
    this->vertices_   = std::move(other.vertices_);
    this->tags_       = std::move(other.tags_);
  }

  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle)
    : handle_ptr_(&handle),
      vertices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
      tags_(std::byte{0})
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle)
    : handle_ptr_(&handle),
      vertices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
      tags_(rmm::device_uvector<tag_t>(0, handle.get_stream()))
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& vertices)
    : handle_ptr_(&handle), vertices_(std::move(vertices)), tags_(std::byte{0})
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle,
               rmm::device_uvector<vertex_t>&& vertices,
               rmm::device_uvector<tag_t>&& tags)
    : handle_ptr_(&handle), vertices_(std::move(vertices)), tags_(std::move(tags))
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle, raft::device_span<vertex_t const> vertices)
    : handle_ptr_(&handle), vertices_(vertices), tags_(std::byte{0})
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle,
               raft::device_span<vertex_t const> vertices,
               raft::device_span<tag_t const> tags)
    : handle_ptr_(&handle), vertices_(vertices), tags_(tags)
  {
  }

  ~key_bucket_t() = default;

  key_bucket_t& operator=(key_bucket_t const& other) = delete;

  // The default move operator should work fine but to silence a compiler warning (CUDA 12.8)
  key_bucket_t& operator=(key_bucket_t&& other)
  {
    if (this != &other) {
      this->handle_ptr_ = other.handle_ptr_;
      this->vertices_   = std::move(other.vertices_);
      this->tags_       = std::move(other.tags_);
    }
    return *this;
  }

  /**
   * @ brief insert a vertex to the bucket
   *
   * @param vertex vertex to insert
   */
  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  void insert(vertex_t vertex)
  {
    CUGRAPH_EXPECTS(vertices_.index() == 1,
                    "insert() is supported only when this bucket holds an owning container.");
    if (std::get<1>(vertices_).size() > 0) {
      rmm::device_scalar<vertex_t> tmp(vertex, handle_ptr_->get_stream());
      insert(tmp.data(), tmp.data() + 1);
    } else {
      std::get<1>(vertices_).resize(1, handle_ptr_->get_stream());
      raft::update_device(
        std::get<1>(vertices_).data(), &vertex, size_t{1}, handle_ptr_->get_stream());
    }
  }

  /**
   * @ brief insert a (vertex, tag) pair to the bucket
   *
   * @param vertex vertex of the (vertex, tag) pair to insert
   * @param tag tag of the (vertex, tag) pair to insert
   */
  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  void insert(thrust::tuple<vertex_t, tag_type> key)
  {
    CUGRAPH_EXPECTS(vertices_.index() == 1,
                    "insert() is supported only when this bucket holds an owning container.");
    if (std::get<1>(vertices_).size() > 0) {
      rmm::device_scalar<vertex_t> tmp_vertex(thrust::get<0>(key), handle_ptr_->get_stream());
      rmm::device_scalar<tag_t> tmp_tag(thrust::get<1>(key), handle_ptr_->get_stream());
      auto pair_first = thrust::make_zip_iterator(tmp_vertex.data(), tmp_tag.data());
      insert(pair_first, pair_first + 1);
    } else {
      std::get<1>(vertices_).resize(1, handle_ptr_->get_stream());
      std::get<1>(tags_).resize(1, handle_ptr_->get_stream());
      auto pair_first =
        thrust::make_zip_iterator(std::get<1>(vertices_).begin(), std::get<1>(tags_).begin());
      thrust::fill(handle_ptr_->get_thrust_policy(), pair_first, pair_first + 1, key);
    }
  }

  /**
   * @ brief insert a list of vertices to the bucket
   *
   * @param vertex_first Iterator pointing to the first (inclusive) element of the vertices stored
   * in device memory.
   * @param vertex_last Iterator pointing to the last (exclusive) element of the vertices stored in
   * device memory.
   */
  template <typename VertexIterator,
            typename tag_type                                 = tag_t,
            std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  void insert(VertexIterator vertex_first, VertexIterator vertex_last)
  {
    static_assert(
      std::is_same_v<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>);

    CUGRAPH_EXPECTS(vertices_.index() == 1,
                    "insert() is supported only when this bucket holds an owning container.");
    if (std::get<1>(vertices_).size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_vertices(
          std::get<1>(vertices_).size() + cuda::std::distance(vertex_first, vertex_last),
          handle_ptr_->get_stream());
        thrust::merge(handle_ptr_->get_thrust_policy(),
                      std::get<1>(vertices_).begin(),
                      std::get<1>(vertices_).end(),
                      vertex_first,
                      vertex_last,
                      merged_vertices.begin());
        merged_vertices.resize(cuda::std::distance(merged_vertices.begin(),
                                                   thrust::unique(handle_ptr_->get_thrust_policy(),
                                                                  merged_vertices.begin(),
                                                                  merged_vertices.end())),
                               handle_ptr_->get_stream());
        std::get<1>(vertices_) = std::move(merged_vertices);
      } else {
        auto cur_size = std::get<1>(vertices_).size();
        std::get<1>(vertices_).resize(cur_size + cuda::std::distance(vertex_first, vertex_last),
                                      handle_ptr_->get_stream());
        thrust::copy(handle_ptr_->get_thrust_policy(),
                     vertex_first,
                     vertex_last,
                     std::get<1>(vertices_).begin() + cur_size);
      }
    } else {
      std::get<1>(vertices_).resize(cuda::std::distance(vertex_first, vertex_last),
                                    handle_ptr_->get_stream());
      thrust::copy(handle_ptr_->get_thrust_policy(),
                   vertex_first,
                   vertex_last,
                   std::get<1>(vertices_).begin());
    }
  }

  /**
   * @ brief insert a list of (vertex, tag) pairs to the bucket
   *
   * @param key_first Iterator pointing to the first (inclusive) element of the (vertex,tag) pairs
   * stored in device memory.
   * @param key_last Iterator pointing to the last (exclusive) element of the  (vertex,tag) pairs
   * stored in device memory.
   */
  template <typename KeyIterator,
            typename tag_type                                  = tag_t,
            std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  void insert(KeyIterator key_first, KeyIterator key_last)
  {
    static_assert(std::is_same_v<typename std::iterator_traits<KeyIterator>::value_type,
                                 thrust::tuple<vertex_t, tag_t>>);

    CUGRAPH_EXPECTS(vertices_.index() == 1,
                    "insert() is supported only when this bucket holds an owning container.");
    if (std::get<1>(vertices_).size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_vertices(
          std::get<1>(vertices_).size() + cuda::std::distance(key_first, key_last),
          handle_ptr_->get_stream());
        rmm::device_uvector<tag_t> merged_tags(merged_vertices.size(), handle_ptr_->get_stream());
        auto old_pair_first =
          thrust::make_zip_iterator(std::get<1>(vertices_).begin(), std::get<1>(tags_).begin());
        auto merged_pair_first =
          thrust::make_zip_iterator(merged_vertices.begin(), merged_tags.begin());
        thrust::merge(handle_ptr_->get_thrust_policy(),
                      old_pair_first,
                      old_pair_first + std::get<1>(vertices_).size(),
                      key_first,
                      key_last,
                      merged_pair_first);
        merged_vertices.resize(
          cuda::std::distance(merged_pair_first,
                              thrust::unique(handle_ptr_->get_thrust_policy(),
                                             merged_pair_first,
                                             merged_pair_first + merged_vertices.size())),
          handle_ptr_->get_stream());
        merged_tags.resize(merged_vertices.size(), handle_ptr_->get_stream());
        std::get<1>(vertices_) = std::move(merged_vertices);
        std::get<1>(tags_)     = std::move(merged_tags);
      } else {
        auto cur_size = std::get<1>(vertices_).size();
        std::get<1>(vertices_).resize(cur_size + cuda::std::distance(key_first, key_last),
                                      handle_ptr_->get_stream());
        std::get<1>(tags_).resize(std::get<1>(vertices_).size(), handle_ptr_->get_stream());
        thrust::copy(
          handle_ptr_->get_thrust_policy(),
          key_first,
          key_last,
          thrust::make_zip_iterator(std::get<1>(vertices_).begin(), std::get<1>(tags_).begin()) +
            cur_size);
      }
    } else {
      std::get<1>(vertices_).resize(cuda::std::distance(key_first, key_last),
                                    handle_ptr_->get_stream());
      std::get<1>(tags_).resize(cuda::std::distance(key_first, key_last),
                                handle_ptr_->get_stream());
      thrust::copy(
        handle_ptr_->get_thrust_policy(),
        key_first,
        key_last,
        thrust::make_zip_iterator(std::get<1>(vertices_).begin(), std::get<1>(tags_).begin()));
    }
  }

  size_t size() const
  {
    return vertices_.index() == 0 ? std::get<0>(vertices_).size() : std::get<1>(vertices_).size();
  }

  template <bool do_aggregate = multi_gpu>
  std::enable_if_t<do_aggregate, size_t> aggregate_size() const
  {
    return host_scalar_allreduce(
      handle_ptr_->get_comms(),
      vertices_.index() == 0 ? std::get<0>(vertices_).size() : std::get<1>(vertices_).size(),
      raft::comms::op_t::SUM,
      handle_ptr_->get_stream());
  }

  template <bool do_aggregate = multi_gpu>
  std::enable_if_t<!do_aggregate, size_t> aggregate_size() const
  {
    return vertices_.index() == 0 ? std::get<0>(vertices_).size() : std::get<1>(vertices_).size();
  }

  void resize(size_t size)
  {
    CUGRAPH_EXPECTS(vertices_.index() == 1,
                    "resize() is supported only when this bucket holds an owning container.");
    std::get<1>(vertices_).resize(size, handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) {
      std::get<1>(tags_).resize(size, handle_ptr_->get_stream());
    }
  }

  void clear() { resize(0); }

  void shrink_to_fit()
  {
    CUGRAPH_EXPECTS(
      vertices_.index() == 1,
      "shrink_to_fit() is supported only when this bucket holds an owning container.");
    std::get<1>(vertices_).shrink_to_fit(handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) {
      std::get<1>(tags_).shrink_to_fit(handle_ptr_->get_stream());
    }
  }

  auto begin()
  {
    CUGRAPH_EXPECTS(
      vertices_.index() == 1,
      "non-const begin() is supported only when this bucket holds an owning container.");
    if constexpr (std::is_same_v<tag_t, void>) {
      return std::get<1>(vertices_).begin();
    } else {
      return thrust::make_zip_iterator(std::get<1>(vertices_).begin(), std::get<1>(tags_).begin());
    }
  }

  auto const cbegin() const
  {
    if constexpr (std::is_same_v<tag_t, void>) {
      return vertices_.index() == 0 ? std::get<0>(vertices_).begin()
                                    : std::get<1>(vertices_).begin();
    } else {
      return vertices_.index() == 0 ? thrust::make_zip_iterator(std::get<0>(vertices_).begin(),
                                                                std::get<0>(tags_).begin())
                                    : thrust::make_zip_iterator(std::get<1>(vertices_).begin(),
                                                                std::get<1>(tags_).begin());
    }
  }

  auto const begin() const { return cbegin(); }

  auto end()
  {
    CUGRAPH_EXPECTS(
      vertices_.index() == 1,
      "non-const end() is supported only when this bucket holds an owning container.");
    return begin() + std::get<1>(vertices_).size();
  }

  auto const cend() const
  {
    return begin() +
           (vertices_.index() == 0 ? std::get<0>(vertices_).size() : std::get<1>(vertices_).size());
  }

  auto const end() const { return cend(); }

  auto vertex_begin()
  {
    CUGRAPH_EXPECTS(
      vertices_.index() == 1,
      "non-const vertex_begin() is supported only when this bucket holds an owning container.");
    return std::get<1>(vertices_).begin();
  }

  auto const vertex_cbegin() const
  {
    return vertices_.index() == 0 ? std::get<0>(vertices_).begin() : std::get<1>(vertices_).begin();
  }

  auto const vertex_begin() const { return vertex_cbegin(); }

  auto vertex_end()
  {
    CUGRAPH_EXPECTS(
      vertices_.index() == 1,
      "non-const vertex_end() is supported only when this bucket holds an owning container.");
    return std::get<1>(vertices_).end();
  }

  auto const vertex_cend() const
  {
    return vertices_.index() == 0 ? std::get<0>(vertices_).end() : std::get<1>(vertices_).end();
  }

  auto const vertex_end() const { return vertex_cend(); }

  bool is_owning() { return (vertices_.index() == 1); }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  std::variant<raft::device_span<vertex_t const>, rmm::device_uvector<vertex_t>> vertices_{};
  optional_variant_type tags_{};
};

template <typename vertex_t,
          typename tag_t                = void,
          bool multi_gpu                = false,
          bool sorted_unique_key_bucket = false>
class vertex_frontier_t {
  static_assert(std::is_same_v<tag_t, void> || std::is_arithmetic_v<tag_t>);

 public:
  using key_type =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  vertex_frontier_t(raft::handle_t const& handle, size_t num_buckets) : handle_ptr_(&handle)
  {
    buckets_.reserve(num_buckets);
    for (size_t i = 0; i < num_buckets; ++i) {
      buckets_.emplace_back(handle);
    }
  }

  size_t num_buckets() const { return buckets_.size(); }

  key_bucket_t<vertex_t, tag_t, multi_gpu, sorted_unique_key_bucket>& bucket(size_t bucket_idx)
  {
    return buckets_[bucket_idx];
  }

  key_bucket_t<vertex_t, tag_t, multi_gpu, sorted_unique_key_bucket> const& bucket(
    size_t bucket_idx) const
  {
    return buckets_[bucket_idx];
  }

  void swap_buckets(size_t bucket_idx0, size_t bucket_idx1)
  {
    std::swap(buckets_[bucket_idx0], buckets_[bucket_idx1]);
  }

  template <typename SplitOp>
  void split_bucket(size_t this_bucket_idx,
                    raft::host_span<size_t const> move_to_bucket_indices,
                    SplitOp split_op)
  {
    auto& this_bucket = bucket(this_bucket_idx);
    if (this_bucket.size() == 0) { return; }

    // 1. apply split_op to each bucket element

    CUGRAPH_EXPECTS(buckets_.size() <= std::numeric_limits<uint8_t>::max(),
                    "Invalid input arguments: the current implementation assumes that bucket "
                    "indices can be represented with uint8_t.");

    rmm::device_uvector<uint8_t> bucket_indices(this_bucket.size(), handle_ptr_->get_stream());
    thrust::transform(
      handle_ptr_->get_thrust_policy(),
      this_bucket.begin(),
      this_bucket.end(),
      bucket_indices.begin(),
      [split_op] __device__(auto key) {
        auto split_op_result = split_op(key);
        return static_cast<uint8_t>(split_op_result ? *split_op_result : kInvalidBucketIdx);
      });

    // 2. separte the elements to stay in this bucket from the elements to be moved to other buckets

    auto pair_first = thrust::make_zip_iterator(bucket_indices.begin(), this_bucket.begin());
    auto new_this_bucket_size = static_cast<size_t>(cuda::std::distance(
      pair_first,
      thrust::stable_partition(  // stable_partition to maintain sorted order within each bucket
        handle_ptr_->get_thrust_policy(),
        pair_first,
        pair_first + bucket_indices.size(),
        [this_bucket_idx = static_cast<uint8_t>(this_bucket_idx)] __device__(auto pair) {
          return thrust::get<0>(pair) == this_bucket_idx;
        })));

    // 3. remove elements with the invalid bucket indices

    bucket_indices.resize(
      new_this_bucket_size +
        cuda::std::distance(pair_first + new_this_bucket_size,
                            thrust::remove_if(handle_ptr_->get_thrust_policy(),
                                              pair_first + new_this_bucket_size,
                                              pair_first + bucket_indices.size(),
                                              [] __device__(auto pair) {
                                                return thrust::get<0>(pair) ==
                                                       static_cast<uint8_t>(kInvalidBucketIdx);
                                              })),
      handle_ptr_->get_stream());
    this_bucket.resize(bucket_indices.size());

    // 4. insert to target buckets and resize this bucket

    insert_to_buckets(bucket_indices.begin() + new_this_bucket_size,
                      bucket_indices.end(),
                      this_bucket.begin() + new_this_bucket_size,
                      move_to_bucket_indices);

    this_bucket.resize(new_this_bucket_size);
    this_bucket.shrink_to_fit();
  }

  template <typename KeyIterator>
  void insert_to_buckets(uint8_t* bucket_idx_first /* [INOUT] */,
                         uint8_t* bucket_idx_last /* [INOUT] */,
                         KeyIterator key_first /* [INOUT] */,
                         raft::host_span<size_t const> to_bucket_indices)
  {
    // 1. group the elements by their target bucket indices

    auto pair_first = thrust::make_zip_iterator(bucket_idx_first, key_first);
    auto pair_last  = pair_first + cuda::std::distance(bucket_idx_first, bucket_idx_last);

    std::vector<size_t> insert_bucket_indices{};
    std::vector<size_t> insert_offsets{};
    std::vector<size_t> insert_sizes{};
    if (to_bucket_indices.size() == 1) {
      insert_bucket_indices =
        std::vector<size_t>(to_bucket_indices.begin(), to_bucket_indices.end());
      insert_offsets = {0};
      insert_sizes   = {static_cast<size_t>(cuda::std::distance(pair_first, pair_last))};
    } else if (to_bucket_indices.size() == 2) {
      auto next_bucket_size = static_cast<size_t>(cuda::std::distance(
        pair_first,
        thrust::stable_partition(  // stable_partition to maintain sorted order within each bucket
          handle_ptr_->get_thrust_policy(),
          pair_first,
          pair_last,
          [next_bucket_idx = static_cast<uint8_t>(to_bucket_indices[0])] __device__(auto pair) {
            return thrust::get<0>(pair) == next_bucket_idx;
          })));
      insert_bucket_indices =
        std::vector<size_t>(to_bucket_indices.begin(), to_bucket_indices.end());
      insert_offsets = {0, next_bucket_size};
      insert_sizes   = {
        next_bucket_size,
        static_cast<size_t>(cuda::std::distance(pair_first + next_bucket_size, pair_last))};
    } else {
      thrust::stable_sort(  // stable_sort to maintain sorted order within each bucket
        handle_ptr_->get_thrust_policy(),
        pair_first,
        pair_last,
        [] __device__(auto lhs, auto rhs) { return thrust::get<0>(lhs) < thrust::get<0>(rhs); });
      rmm::device_uvector<uint8_t> d_indices(to_bucket_indices.size(), handle_ptr_->get_stream());
      rmm::device_uvector<size_t> d_counts(d_indices.size(), handle_ptr_->get_stream());
      // FIXME: thrust::lower_bound & thrust::upper_bound will be faster
      auto it = thrust::reduce_by_key(handle_ptr_->get_thrust_policy(),
                                      bucket_idx_first,
                                      bucket_idx_last,
                                      thrust::make_constant_iterator(size_t{1}),
                                      d_indices.begin(),
                                      d_counts.begin());
      d_indices.resize(cuda::std::distance(d_indices.begin(), thrust::get<0>(it)),
                       handle_ptr_->get_stream());
      d_counts.resize(d_indices.size(), handle_ptr_->get_stream());
      std::vector<uint8_t> h_indices(d_indices.size());
      std::vector<size_t> h_counts(h_indices.size());
      raft::update_host(
        h_indices.data(), d_indices.data(), d_indices.size(), handle_ptr_->get_stream());
      raft::update_host(
        h_counts.data(), d_counts.data(), d_counts.size(), handle_ptr_->get_stream());
      handle_ptr_->sync_stream();

      size_t offset{0};
      for (size_t i = 0; i < h_indices.size(); ++i) {
        insert_bucket_indices[i] = static_cast<size_t>(h_indices[i]);
        insert_offsets[i]        = offset;
        insert_sizes[i]          = h_counts[i];
        offset += insert_sizes[i];
      }
    }

    // 2. insert to the target buckets

    for (size_t i = 0; i < insert_offsets.size(); ++i) {
      bucket(insert_bucket_indices[i])
        .insert(key_first + insert_offsets[i], key_first + (insert_offsets[i] + insert_sizes[i]));
    }
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  std::vector<key_bucket_t<vertex_t, tag_t, multi_gpu, sorted_unique_key_bucket>> buckets_{};
};

}  // namespace cugraph

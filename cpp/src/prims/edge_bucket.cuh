/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
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
#include <vector>

namespace cugraph {

// key type is either thrust::tuple<vertex_t, vertex_t> (tag_t == void) or thrust::tuple<vertex_t,
// vertex_t, tag_t> (tag_t != void). tag_t can be used to point a specific edge if there are
// multiple edges between a source and a destination (e.g. tag_t can be an edge ID type). If
// sorted_unique is true, stores unique key objects in the sorted (non-descending) order. If false,
// there can be duplicates and the elements may not be sorted. Use source as the primary key and
// destination as the secondary key for sorting if src_major is true. Use destination as the primary
// key and source as the secondary key if src_major is false. If tag_t is not void, use tag as the
// tertiary key in sorting.
template <typename vertex_t,
          typename tag_t     = void,
          bool src_major     = false,
          bool multi_gpu     = false,
          bool sorted_unique = false>
class edge_bucket_t {
 public:
  using key_type = std::conditional_t<std::is_same_v<tag_t, void>,
                                      thrust::tuple<vertex_t, vertex_t>,
                                      thrust::tuple<vertex_t, vertex_t, tag_t>>;

  static bool constexpr is_src_major     = src_major;
  static bool constexpr is_sorted_unique = sorted_unique;

  static_assert(std::is_same_v<tag_t, void> || std::is_arithmetic_v<tag_t>);

  using optional_buffer_type = std::
    conditional_t<std::is_same_v<tag_t, void>, std::byte /* dummy */, rmm::device_uvector<tag_t>>;

  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  edge_bucket_t(raft::handle_t const& handle)
    : handle_ptr_(&handle),
      majors_(0, handle.get_stream()),
      minors_(0, handle.get_stream()),
      tags_(std::byte{0})
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  edge_bucket_t(raft::handle_t const& handle)
    : handle_ptr_(&handle),
      majors_(0, handle.get_stream()),
      minors_(0, handle.get_stream()),
      tags_(0, handle.get_stream())
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  edge_bucket_t(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>&& srcs,
                rmm::device_uvector<vertex_t>&& dsts)
    : handle_ptr_(&handle),
      majors_(std::move(src_major ? srcs : dsts)),
      minors_(std::move(src_major ? dsts : srcs)),
      tags_(std::byte{0})
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  edge_bucket_t(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>&& srcs,
                rmm::device_uvector<vertex_t>&& dsts,
                rmm::device_uvector<tag_t>&& tags)
    : handle_ptr_(&handle),
      majors_(std::move(src_major ? srcs : dsts)),
      minors_(std::move(src_major ? dsts : srcs)),
      tags_(std::move(tags))
  {
  }

  /**
   * @ brief insert an edge to the bucket
   *
   * @param src edge source vertex.
   * @param dst edge destination vertex.
   */
  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  void insert(vertex_t src, vertex_t dst)
  {
    if (majors_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp_src(src, handle_ptr_->get_stream());
      rmm::device_scalar<vertex_t> tmp_dst(dst, handle_ptr_->get_stream());
      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(tmp_src.data(), tmp_dst.data()));
      insert(tmp_src.data(), tmp_src.data() + 1, tmp_dst.data());
    } else {
      auto major = src_major ? src : dst;
      auto minor = src_major ? dst : src;
      majors_.resize(1, handle_ptr_->get_stream());
      minors_.resize(1, handle_ptr_->get_stream());
      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(majors_.data(), minors_.data()));
      thrust::fill(handle_ptr_->get_thrust_policy(),
                   pair_first,
                   pair_first + 1,
                   thrust::make_tuple(major, minor));
    }
  }

  /**
   * @ brief insert a tagged-edge to the bucket
   *
   * @param src edge source vertex.
   * @param dst edge destination vertex.
   * @param tag edge tag.
   */
  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  void insert(vertex_t src, vertex_t dst, tag_type tag)
  {
    if (majors_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp_src(src, handle_ptr_->get_stream());
      rmm::device_scalar<vertex_t> tmp_dst(dst, handle_ptr_->get_stream());
      rmm::device_scalar<tag_t> tmp_tag(tag, handle_ptr_->get_stream());
      auto triplet_first = thrust::make_zip_iterator(
        thrust::make_tuple(tmp_src.data(), tmp_dst.data(), tmp_tag.data()));
      insert(tmp_src.data(), tmp_src.data() + 1, tmp_dst.data(), tmp_tag.data());
    } else {
      auto major = src_major ? src : dst;
      auto minor = src_major ? dst : src;
      majors_.resize(1, handle_ptr_->get_stream());
      minors_.resize(1, handle_ptr_->get_stream());
      tags_.resize(1, handle_ptr_->get_stream());
      auto triplet_first =
        thrust::make_zip_iterator(thrust::make_tuple(majors_.data(), minors_.data(), tags_.data()));
      thrust::fill(handle_ptr_->get_thrust_policy(),
                   triplet_first,
                   triplet_first + 1,
                   thrust::make_tuple(major, minor, tag));
    }
  }

  /**
   * @ brief insert a list of edges to the bucket
   *
   * @param src_first Iterator pointing to the first (inclusive) element of the edge source vertices
   * in device memory.
   * @param src_last Iterator pointing to the last (exclusive) element of the edge source vertices
   * stored in device memory.
   * @param dst_first Iterator pointing to the first (inclusive) element of the edge destination
   * vertices in device memory.
   */
  template <typename VertexIterator,
            typename tag_type                                 = tag_t,
            std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  void insert(VertexIterator src_first, VertexIterator src_last, VertexIterator dst_first)
  {
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<VertexIterator>::value_type, vertex_t>);

    auto major_first    = src_major ? src_first : dst_first;
    auto major_last     = major_first + cuda::std::distance(src_first, src_last);
    auto minor_first    = src_major ? dst_first : src_first;
    auto new_pair_first = thrust::make_zip_iterator(thrust::make_tuple(major_first, minor_first));

    if (majors_.size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_majors(
          majors_.size() + cuda::std::distance(major_first, major_last), handle_ptr_->get_stream());
        rmm::device_uvector<vertex_t> merged_minors(merged_majors.size(),
                                                    handle_ptr_->get_stream());
        auto old_pair_first =
          thrust::make_zip_iterator(thrust::make_tuple(majors_.begin(), minors_.begin()));
        auto merged_pair_first = thrust::make_zip_iterator(
          thrust::make_tuple(merged_majors.begin(), merged_minors.begin()));
        thrust::merge(handle_ptr_->get_thrust_policy(),
                      old_pair_first,
                      old_pair_first + majors_.size(),
                      new_pair_first,
                      new_pair_first + cuda::std::distance(major_first, major_last),
                      merged_pair_first);
        merged_majors.resize(
          cuda::std::distance(merged_pair_first,
                              thrust::unique(handle_ptr_->get_thrust_policy(),
                                             merged_pair_first,
                                             merged_pair_first + merged_majors.size())),
          handle_ptr_->get_stream());
        merged_minors.resize(merged_majors.size(), handle_ptr_->get_stream());
        merged_majors.shrink_to_fit(handle_ptr_->get_stream());
        merged_minors.shrink_to_fit(handle_ptr_->get_stream());
        majors_ = std::move(merged_majors);
        minors_ = std::move(merged_minors);
      } else {
        auto cur_size = majors_.size();
        majors_.resize(cur_size + cuda::std::distance(major_first, major_last),
                       handle_ptr_->get_stream());
        minors_.resize(majors_.size(), handle_ptr_->get_stream());
        thrust::copy(
          handle_ptr_->get_thrust_policy(),
          new_pair_first,
          new_pair_first + cuda::std::distance(major_first, major_last),
          thrust::make_zip_iterator(thrust::make_tuple(majors_.begin(), minors_.begin())) +
            cur_size);
      }
    } else {
      majors_.resize(cuda::std::distance(major_first, major_last), handle_ptr_->get_stream());
      minors_.resize(majors_.size(), handle_ptr_->get_stream());
      thrust::copy(handle_ptr_->get_thrust_policy(),
                   new_pair_first,
                   new_pair_first + cuda::std::distance(major_first, major_last),
                   thrust::make_zip_iterator(thrust::make_tuple(majors_.begin(), minors_.begin())));
    }
  }

  /**
   * @ brief insert a list of tagged-edges to the bucket
   *
   * @param src_first Iterator pointing to the first (inclusive) element of the edge source vertices
   * in device memory.
   * @param src_last Iterator pointing to the last (exclusive) element of the edge source vertices
   * stored in device memory.
   * @param dst_first Iterator pointing to the first (inclusive) element of the edge destination
   * vertices in device memory.
   * @param tag_first Iterator pointing to the first (inclusive) element of the edge tags in device
   * memory.
   */
  template <typename VertexIterator,
            typename TagIterator,
            typename tag_type                                  = tag_t,
            std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  void insert(VertexIterator src_first,
              VertexIterator src_last,
              VertexIterator dst_first,
              TagIterator tag_first)
  {
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<VertexIterator>::value_type, vertex_t>);
    static_assert(std::is_same_v<typename thrust::iterator_traits<TagIterator>::value_type, tag_t>);

    auto major_first = src_major ? src_first : dst_first;
    auto major_last  = major_first + cuda::std::distance(src_first, src_last);
    auto minor_first = src_major ? dst_first : src_first;
    auto new_triplet_first =
      thrust::make_zip_iterator(thrust::make_tuple(major_first, minor_first, tag_first));

    if (majors_.size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_majors(
          majors_.size() + cuda::std::distance(major_first, major_last), handle_ptr_->get_stream());
        rmm::device_uvector<vertex_t> merged_minors(merged_majors.size(),
                                                    handle_ptr_->get_stream());
        rmm::device_uvector<tag_t> merged_tags(merged_majors.size(), handle_ptr_->get_stream());
        auto old_triplet_first = thrust::make_zip_iterator(
          thrust::make_tuple(majors_.begin(), minors_.begin(), tags_.begin()));
        auto merged_triplet_first = thrust::make_zip_iterator(
          thrust::make_tuple(merged_majors.begin(), merged_minors.begin(), merged_tags.begin()));
        thrust::merge(handle_ptr_->get_thrust_policy(),
                      old_triplet_first,
                      old_triplet_first + majors_.size(),
                      new_triplet_first,
                      new_triplet_first + cuda::std::distance(major_first, major_last),
                      merged_triplet_first);
        merged_majors.resize(
          cuda::std::distance(merged_triplet_first,
                              thrust::unique(handle_ptr_->get_thrust_policy(),
                                             merged_triplet_first,
                                             merged_triplet_first + merged_majors.size())),
          handle_ptr_->get_stream());
        merged_minors.resize(merged_majors.size(), handle_ptr_->get_stream());
        merged_tags.resize(merged_majors.size(), handle_ptr_->get_stream());
        merged_majors.shrink_to_fit(handle_ptr_->get_stream());
        merged_minors.shrink_to_fit(handle_ptr_->get_stream());
        merged_tags.shrink_to_fit(handle_ptr_->get_stream());
        majors_ = std::move(merged_majors);
        minors_ = std::move(merged_minors);
        tags_   = std::move(merged_tags);
      } else {
        auto cur_size = majors_.size();
        majors_.resize(cur_size + cuda::std::distance(major_first, major_last),
                       handle_ptr_->get_stream());
        minors_.resize(majors_.size(), handle_ptr_->get_stream());
        tags_.resize(majors_.size(), handle_ptr_->get_stream());
        thrust::copy(handle_ptr_->get_thrust_policy(),
                     new_triplet_first,
                     new_triplet_first + cuda::std::distance(major_first, major_last),
                     thrust::make_zip_iterator(
                       thrust::make_tuple(majors_.begin(), minors_.begin(), tags_.begin())) +
                       cur_size);
      }
    } else {
      majors_.resize(cuda::std::distance(major_first, major_last), handle_ptr_->get_stream());
      minors_.resize(majors_.size(), handle_ptr_->get_stream());
      tags_.resize(majors_.size(), handle_ptr_->get_stream());
      thrust::copy(handle_ptr_->get_thrust_policy(),
                   new_triplet_first,
                   new_triplet_first + cuda::std::distance(major_first, major_last),
                   thrust::make_zip_iterator(
                     thrust::make_tuple(majors_.begin(), minors_.begin(), tags_.begin())));
    }
  }

  size_t size() const { return majors_.size(); }

  template <bool do_aggregate = multi_gpu>
  std::enable_if_t<do_aggregate, size_t> aggregate_size() const
  {
    return host_scalar_allreduce(
      handle_ptr_->get_comms(), majors_.size(), raft::comms::op_t::SUM, handle_ptr_->get_stream());
  }

  template <bool do_aggregate = multi_gpu>
  std::enable_if_t<!do_aggregate, size_t> aggregate_size() const
  {
    return majors_.size();
  }

  void resize(size_t size)
  {
    majors_.resize(size, handle_ptr_->get_stream());
    minors_.resize(size, handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) { tags_.resize(size, handle_ptr_->get_stream()); }
  }

  void clear() { resize(0); }

  void shrink_to_fit()
  {
    majors_.shrink_to_fit(handle_ptr_->get_stream());
    minors_.shrink_to_fit(handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) { tags_.shrink_to_fit(handle_ptr_->get_stream()); }
  }

  auto const src_begin() const { return src_major ? majors_.begin() : minors_.begin(); }

  auto src_begin() { return src_major ? majors_.begin() : minors_.begin(); }

  auto const src_end() const { return (src_major ? majors_.begin() : minors_.begin()) + size(); }

  auto src_end() { return (src_major ? majors_.begin() : minors_.begin()) + size(); }

  auto const dst_begin() const { return src_major ? minors_.begin() : majors_.begin(); }

  auto dst_begin() { return src_major ? minors_.begin() : majors_.begin(); }

  auto const dst_end() const { return (src_major ? minors_.begin() : majors_.begin()) + size(); }

  auto dst_end() { return (src_major ? minors_.begin() : majors_.begin()) + size(); }

  auto const tag_begin() const { return tags_.begin(); }

  auto tag_begin() { return tags_.begin(); }

  auto const tag_end() const { return tags_.begin() + size(); }

  auto tag_end() { tags_.begin() + size(); }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_uvector<vertex_t> majors_;
  rmm::device_uvector<vertex_t> minors_;
  optional_buffer_type tags_;
};

}  // namespace cugraph

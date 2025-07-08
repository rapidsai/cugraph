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

// edges are uniquely indexed by either source and destination vertex ID pairs or source,
// destination, and multi-edge index triplets; the latter for multi-graphs.
// If sorted_unique is true, stores unique keys in the sorted (non-descending) order. If false,
// there can be duplicates and the keys may not be sorted. Use source as the primary key and
// destination as the secondary key for sorting if src_major is true. Use destination as the primary
// key and source as the secondary key if src_major is false. Multi-edge indices (if releveant) are
// used as the tertiary keys.
template <typename vertex_t,
          typename edge_t,
          bool src_major     = false,
          bool multi_gpu     = false,
          bool sorted_unique = false>
class edge_bucket_t {
 public:
  static bool constexpr is_src_major     = src_major;
  static bool constexpr is_sorted_unique = sorted_unique;

  edge_bucket_t(raft::handle_t const& handle, bool multigraph)
    : handle_ptr_(&handle),
      majors_(0, handle.get_stream()),
      minors_(0, handle.get_stream()),
      multi_edge_indices_(std::nullopt)
  {
    if (multigraph) { multi_edge_indices_ = rmm::device_uvector<edge_t>(0, handle.get_stream()); }
  }

  edge_bucket_t(raft::handle_t const& handle,
                rmm::device_uvector<vertex_t>&& srcs,
                rmm::device_uvector<vertex_t>&& dsts,
                std::optional<rmm::device_uvector<edge_t>>&& multi_edge_indices)
    : handle_ptr_(&handle),
      majors_(std::move(src_major ? srcs : dsts)),
      minors_(std::move(src_major ? dsts : srcs)),
      multi_edge_indices_(std::move(multi_edge_indices))
  {
  }

  /**
   * @ brief insert an edge to the bucket
   *
   * @param src edge source vertex.
   * @param dst edge destination vertex.
   * @param multi_edge_index multi-edge index (for multi-graphs).
   */
  void insert(vertex_t src, vertex_t dst, std::optional<edge_t> multi_edge_index)
  {
    CUGRAPH_EXPECTS(multi_edge_indices_.has_value() == multi_edge_index.has_value(),
                    "Invalid input argument: multi_edge_index.has_value() does not match with "
                    "multi_edge_indices_.has_value()");

    if (majors_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp_src(src, handle_ptr_->get_stream());
      rmm::device_scalar<vertex_t> tmp_dst(dst, handle_ptr_->get_stream());
      std::optional<rmm::device_scalar<edge_t>> tmp_multi_edge_index{std::nullopt};
      if (multi_edge_index) {
        tmp_multi_edge_index =
          rmm::device_scalar<edge_t>(*multi_edge_index, handle_ptr_->get_stream());
      }
      insert(
        tmp_src.data(),
        tmp_src.data() + 1,
        tmp_dst.data(),
        tmp_multi_edge_index ? std::make_optional(tmp_multi_edge_index->data()) : std::nullopt);
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
      if (multi_edge_index) {
        multi_edge_indices_->resize(1, handle_ptr_->get_stream());
        thrust::fill(handle_ptr_->get_thrust_policy(),
                     multi_edge_indices_->begin(),
                     multi_edge_indices_->begin() + 1,
                     *multi_edge_index);
      }
    }
  }

  /**
   * @ brief insert a list of edges to the bucket
   *
   * @param src_first Iterator pointing to the first (inclusive) element of the edge source
   * vertices in device memory.
   * @param src_last Iterator pointing to the last (exclusive) element of the edge source vertices
   * stored in device memory.
   * @param dst_first Iterator pointing to the first (inclusive) element of the edge destination
   * vertices in device memory.
   * @param multi_edge_index_first Iterator pointing to the first (inclusive) element of the
   * multi-edge indices in device memory (for multi-graphs).
   */
  template <typename VertexIterator, typename MultiEdgeIndexIterator>
  void insert(VertexIterator src_first,
              VertexIterator src_last,
              VertexIterator dst_first,
              std::optional<MultiEdgeIndexIterator> multi_edge_index_first)
  {
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<VertexIterator>::value_type, vertex_t>);
    static_assert(
      std::is_same_v<typename thrust::iterator_traits<MultiEdgeIndexIterator>::value_type, edge_t>);

    CUGRAPH_EXPECTS(multi_edge_indices_.has_value() == multi_edge_index_first.has_value(),
                    "Invalid input argument: multi_edge_index_first.has_value() does not match "
                    "with multi_edge_indices_.has_value()");

    auto major_first = src_major ? src_first : dst_first;
    auto major_last  = major_first + cuda::std::distance(src_first, src_last);
    auto minor_first = src_major ? dst_first : src_first;

    if (majors_.size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_majors(
          majors_.size() + cuda::std::distance(major_first, major_last), handle_ptr_->get_stream());
        rmm::device_uvector<vertex_t> merged_minors(merged_majors.size(),
                                                    handle_ptr_->get_stream());
        std::optional<rmm::device_uvector<edge_t>> merged_multi_edge_indices =
          multi_edge_index_first ? std::make_optional<rmm::device_uvector<edge_t>>(
                                     merged_majors.size(), handle_ptr_->get_stream())
                                 : std::nullopt;
        size_t new_size{0};
        if (multi_edge_index_first) {
          auto new_triplet_first =
            thrust::make_zip_iterator(major_first, minor_first, *multi_edge_index_first);
          auto old_triplet_first = thrust::make_zip_iterator(
            majors_.begin(), minors_.begin(), multi_edge_indices_->begin());
          auto merged_triplet_first = thrust::make_zip_iterator(
            merged_majors.begin(), merged_minors.begin(), merged_multi_edge_indices->begin());
          thrust::merge(handle_ptr_->get_thrust_policy(),
                        old_triplet_first,
                        old_triplet_first + majors_.size(),
                        new_triplet_first,
                        new_triplet_first + cuda::std::distance(major_first, major_last),
                        merged_triplet_first);
          new_size = static_cast<size_t>(
            cuda::std::distance(merged_triplet_first,
                                thrust::unique(handle_ptr_->get_thrust_policy(),
                                               merged_triplet_first,
                                               merged_triplet_first + merged_majors.size())));
        } else {
          auto new_pair_first =
            thrust::make_zip_iterator(thrust::make_tuple(major_first, minor_first));
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
          new_size = static_cast<size_t>(
            cuda::std::distance(merged_pair_first,
                                thrust::unique(handle_ptr_->get_thrust_policy(),
                                               merged_pair_first,
                                               merged_pair_first + merged_majors.size())));
        }
        merged_minors.resize(new_size, handle_ptr_->get_stream());
        merged_minors.resize(new_size, handle_ptr_->get_stream());
        merged_majors.shrink_to_fit(handle_ptr_->get_stream());
        merged_minors.shrink_to_fit(handle_ptr_->get_stream());
        majors_ = std::move(merged_majors);
        minors_ = std::move(merged_minors);
        if (merged_multi_edge_indices) {
          merged_multi_edge_indices->resize(new_size, handle_ptr_->get_stream());
          merged_multi_edge_indices->shrink_to_fit(handle_ptr_->get_stream());
          multi_edge_indices_ = std::move(merged_multi_edge_indices);
        }
      } else {
        auto new_pair_first =
          thrust::make_zip_iterator(thrust::make_tuple(major_first, minor_first));
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
        if (multi_edge_index_first) {
          multi_edge_indices_->resize(majors_.size(), handle_ptr_->get_stream());
          thrust::copy(handle_ptr_->get_thrust_policy(),
                       *multi_edge_index_first,
                       *multi_edge_index_first + cuda::std::distance(major_first, major_last),
                       multi_edge_indices_->begin() + cur_size);
        }
      }
    } else {
      auto new_pair_first = thrust::make_zip_iterator(thrust::make_tuple(major_first, minor_first));
      majors_.resize(cuda::std::distance(major_first, major_last), handle_ptr_->get_stream());
      minors_.resize(majors_.size(), handle_ptr_->get_stream());
      thrust::copy(handle_ptr_->get_thrust_policy(),
                   new_pair_first,
                   new_pair_first + cuda::std::distance(major_first, major_last),
                   thrust::make_zip_iterator(thrust::make_tuple(majors_.begin(), minors_.begin())));
      if (multi_edge_index_first) {
        multi_edge_indices_->resize(majors_.size(), handle_ptr_->get_stream());
        thrust::copy(handle_ptr_->get_thrust_policy(),
                     *multi_edge_index_first,
                     *multi_edge_index_first + cuda::std::distance(major_first, major_last),
                     multi_edge_indices_->begin());
      }
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
    if (multi_edge_indices_) { multi_edge_indices_->resize(size, handle_ptr_->get_stream()); }
  }

  void clear() { resize(0); }

  void shrink_to_fit()
  {
    majors_.shrink_to_fit(handle_ptr_->get_stream());
    minors_.shrink_to_fit(handle_ptr_->get_stream());
    if (multi_edge_indices_) { multi_edge_indices_->shrink_to_fit(handle_ptr_->get_stream()); }
  }

  auto const src_begin() const { return src_major ? majors_.begin() : minors_.begin(); }

  auto src_begin() { return src_major ? majors_.begin() : minors_.begin(); }

  auto const src_end() const { return (src_major ? majors_.begin() : minors_.begin()) + size(); }

  auto src_end() { return (src_major ? majors_.begin() : minors_.begin()) + size(); }

  vertex_t const* src_data() const { return src_major ? majors_.data() : minors_.data(); }

  vertex_t* src_data() { return src_major ? majors_.data() : minors_.data(); }

  auto const dst_begin() const { return src_major ? minors_.begin() : majors_.begin(); }

  auto dst_begin() { return src_major ? minors_.begin() : majors_.begin(); }

  auto const dst_end() const { return (src_major ? minors_.begin() : majors_.begin()) + size(); }

  auto dst_end() { return (src_major ? minors_.begin() : majors_.begin()) + size(); }

  vertex_t const* dst_data() const { return src_major ? minors_.data() : majors_.data(); }

  vertex_t* dst_data() { return src_major ? minors_.data() : majors_.data(); }

  auto const multi_edge_index_begin() const
  {
    return multi_edge_indices_ ? std::make_optional(multi_edge_indices_->begin()) : std::nullopt;
  }

  auto multi_edge_index_begin()
  {
    return multi_edge_indices_ ? std::make_optional(multi_edge_indices_->begin()) : std::nullopt;
  }

  auto const multi_edge_index_end() const
  {
    return multi_edge_indices_ ? std::make_optional(multi_edge_indices_->begin() + size())
                               : std::nullopt;
  }

  auto multi_edge_index_end()
  {
    return multi_edge_indices_ ? std::make_optional(multi_edge_indices_->begin() + size())
                               : std::nullopt;
  }

  std::optional<edge_t const*> multi_edge_index_data() const
  {
    return multi_edge_indices_ ? std::make_optional(multi_edge_indices_->data()) : std::nullopt;
  }

  std::optional<edge_t*> multi_edge_index_data()
  {
    return multi_edge_indices_ ? std::make_optional(multi_edge_indices_->data()) : std::nullopt;
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_uvector<vertex_t> majors_;
  rmm::device_uvector<vertex_t> minors_;
  std::optional<rmm::device_uvector<edge_t>> multi_edge_indices_{std::nullopt};
};

}  // namespace cugraph

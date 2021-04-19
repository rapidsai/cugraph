/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <utilities/error.hpp>
#include <utilities/host_scalar_comm.cuh>
#include <utilities/thrust_tuple_utils.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cinttypes>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename T>
struct optional_buffer_t {
  decltype(allocate_dataframe_buffer<T>(0, cudaStream_t{nullptr})) buffer;
};

template <>
struct optional_buffer_t<void> {
};

}  // namespace detail

// stores unique vertex_t (tag_t == void) or (vertex_t, tag_t) pair (tag_t != void) objects in the
// sorted (non-descending) order
// this class object can optionally store payload_t objects (payload_t != void) in addition to
// vertex_t (tag_t == void) or (vertex_t, tag_t) pair (tag_t != void) objects.
template <typename vertex_t,
          typename tag_t     = void,
          typename payload_t = void,
          bool is_multi_gpu  = false>
class SortedUniqueElementBucket {
  static_assert(std::is_same_v<tag_t, void> ||
                is_arithmetic_or_thrust_tuple_of_arithmetic<tag_t>::value);

 public:
  SortedUniqueElementBucket(raft::handle_t const& handle)
    : handle_ptr_(&handle), vertices_(0, handle.get_stream())
  {
    if constexpr (!std::is_same<tag_t, void>::value) {
      tags_.buffer = allocate_dataframe_buffer<tag_t>(0, handle.get_stream());
    }
    if constexpr (!std::is_same<payload_t, void>::value) {
      payloads_.buffer = allocate_dataframe_buffer<payload_t>(0, handle.get_stream());
    }
  }

  /**
   * @ brief insert a vertex to the bucket
   *
   * @param vertex vertex to insert
   */
  template <typename tag_type                                     = tag_t,
            typename payload_type                                 = payload_t,
            std::enable_if_t<std::is_same_v<tag_type, void> &&
                             std::is_same_v<payload_type, void>>* = nullptr>
  void insert(vertex_t vertex)
  {
    if (vertices_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp(vertex, handle_ptr_->get_stream());
      insert(tmp.data(), tmp.data() + 1);
    } else {
      vertices_.resize(1, handle_ptr_->get_stream());
      raft::update_device(vertices_.data(), &vertex, size_t{1}, handle_ptr_->get_stream());
    }
  }

  /**
   * @ brief insert a (vertex, tag) pair to the bucket
   *
   * @param vertex vertex of the (vertex, tag) pair to insert
   * @param tag tag of the (vertex, tag) pair to insert
   */
  template <typename tag_type                                     = tag_t,
            typename payload_type                                 = payload_t,
            std::enable_if_t<!std::is_same_v<tag_type, void> &&
                             std::is_same_v<payload_type, void>>* = nullptr>
  void insert(vertex_t vertex, tag_type tag)
  {
    if (vertices_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp(vertex, handle_ptr_->get_stream());
      auto tag_buffer = allocate_dataframe_buffer<tag_type>(1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<tag_type>(tag_buffer),
                   get_dataframe_buffer_end<tag_type>(tag_buffer),
                   tag);
      insert(tmp.data(), tmp.data() + 1, get_dataframe_buffer_begin<tag_type>(tag_buffer));
    } else {
      vertices_.resize(1, handle_ptr_->get_stream());
      raft::update_device(vertices_.data(), &vertex, size_t{1}, handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(tags_.buffer, 1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<tag_type>(tags_.buffer),
                   get_dataframe_buffer_end<tag_type>(tags_.buffer),
                   tag);
    }
  }

  /**
   * @ brief insert a vertex with a payload to the bucket
   *
   * @param vertex vertex to insert
   * @param payload payload to insert with the vertex
   * @param reduce_op Reduction operation to accumulate two payload_t variables, @reduce_op should
   * be a pure function with no side effect.
   */
  template <typename ReduceOp,
            typename tag_type                                      = tag_t,
            typename payload_type                                  = payload_t,
            std::enable_if_t<std::is_same_v<tag_type, void> &&
                             !std::is_same_v<payload_type, void>>* = nullptr>
  void insert(vertex_t vertex, payload_type payload, ReduceOp reduce_op)
  {
    if (vertices_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp(vertex, handle_ptr_->get_stream());
      auto payload_buffer = allocate_dataframe_buffer<payload_type>(1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<payload_type>(payload_buffer),
                   get_dataframe_buffer_end<payload_type>(payload_buffer),
                   payload);
      insert(tmp.data(), tmp.data() + 1, get_dataframe_buffer_begin<payload_type>(payload_buffer));
    } else {
      vertices_.resize(1, handle_ptr_->get_stream());
      raft::update_device(vertices_.data(), &vertex, size_t{1}, handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(payloads_.buffer, 1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<payload_type>(payloads_.buffer),
                   get_dataframe_buffer_end<payload_type>(payloads_.buffer),
                   payload);
    }
  }

  /**
   * @ brief insert a (vertex, tag) pair with a payload to the bucket
   *
   * @param vertex vertex of the (vertex, tag) pair to insert
   * @param tag tag of the (vertex, tag) pair to insert
   * @param payload payload to insert with the (vertex, tag) pair
   * @param reduce_op Reduction operation to accumulate two payload_t variables, @reduce_op should
   */
  template <typename ReduceOp,
            typename tag_type                                      = tag_t,
            typename payload_type                                  = payload_t,
            std::enable_if_t<!std::is_same_v<tag_type, void> &&
                             !std::is_same_v<payload_type, void>>* = nullptr>
  void insert(vertex_t vertex, tag_type tag, payload_type payload, ReduceOp reduce_op)
  {
    if (vertices_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp(vertex, handle_ptr_->get_stream());
      auto tag_buffer = allocate_dataframe_buffer<tag_type>(1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<tag_type>(tag_buffer),
                   get_dataframe_buffer_end<tag_type>(tag_buffer),
                   tag);
      auto payload_buffer = allocate_dataframe_buffer<payload_type>(1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<payload_type>(payload_buffer),
                   get_dataframe_buffer_end<payload_type>(payload_buffer),
                   payload);
      insert(tmp.data(),
             tmp.data() + 1,
             get_dataframe_buffer_begin<tag_type>(tag_buffer),
             get_dataframe_buffer_begin<payload_type>(payload_buffer));
    } else {
      vertices_.resize(1, handle_ptr_->get_stream());
      raft::update_device(vertices_.data(), &vertex, size_t{1}, handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(tags_.buffer, 1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<tag_type>(tags_.buffer),
                   get_dataframe_buffer_end<tag_type>(tags_.buffer),
                   tag);
      resize_dataframe_buffer<payload_type>(payloads_.buffer, 1, handle_ptr_->get_stream());
      thrust::fill(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   get_dataframe_buffer_begin<payload_type>(payloads_.buffer),
                   get_dataframe_buffer_end<payload_type>(payloads_.buffer),
                   payload);
    }
  }

  /**
   * @ brief insert a list of vertices to the bucket
   *
   * @param vertex_first Iterator pointing to the first (inclusive) element of the vertex list
   * stored in device memory.
   * @param vertex_last Iterator pointing to the last (exclusive) element of the vertex list stored
   * in device memory.
   */
  template <typename VertexIterator,
            typename tag_type                                     = tag_t,
            typename payload_type                                 = payload_t,
            std::enable_if_t<std::is_same_v<tag_type, void> &&
                             std::is_same_v<payload_type, void>>* = nullptr>
  void insert(VertexIterator vertex_first, VertexIterator vertex_last)
  {
    static_assert(
      std::is_same_v<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>);

    if (vertices_.size() > 0) {
      rmm::device_uvector<vertex_t> merged_vertices(
        vertices_.size() + thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      thrust::merge(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                    vertices_.begin(),
                    vertices_.end(),
                    vertex_first,
                    vertex_last,
                    merged_vertices.begin());
      merged_vertices.resize(
        thrust::distance(
          merged_vertices.begin(),
          thrust::unique(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                         merged_vertices.begin(),
                         merged_vertices.end())),
        handle_ptr_->get_stream());
      merged_vertices.shrink_to_fit(handle_ptr_->get_stream());
      vertices_ = std::move(merged_vertices);
    } else {
      vertices_.resize(thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      thrust::copy(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   vertex_first,
                   vertex_last,
                   vertices_.begin());
    }
  }

  /**
   * @ brief insert a list of vertices and vertx tags to the bucket
   *
   * @param vertex_first Iterator pointing to the first (inclusive) element of the vertex list
   * stored in device memory.
   * @param vertex_last Iterator pointing to the last (exclusive) element of the vertex list stored
   * in device memory.
   * @param tag_first Iterator pointing to the first (inclusive) element of the vertex tags stored
   * in device memory.
   */
  template <typename VertexIterator,
            typename TagIterator,
            typename tag_type                                     = tag_t,
            typename payload_type                                 = payload_t,
            std::enable_if_t<!std::is_same_v<tag_type, void> &&
                             std::is_same_v<payload_type, void>>* = nullptr>
  void insert(VertexIterator vertex_first, VertexIterator vertex_last, TagIterator tag_first)
  {
    static_assert(
      std::is_same_v<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>);
    static_assert(std::is_same_v<typename std::iterator_traits<TagIterator>::value_type, tag_type>);

    if (vertices_.size() > 0) {
      rmm::device_uvector<vertex_t> merged_vertices(
        vertices_.size() + thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      auto merged_tag_buffer =
        allocate_dataframe_buffer<tag_type>(merged_vertices.size(), handle_ptr_->get_stream());
      auto old_pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(vertices_.begin(), get_dataframe_buffer_begin<tag_type>(tags_.buffer)));
      auto new_pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertex_first, tag_first));
      auto merged_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        merged_vertices.begin(), get_dataframe_buffer_begin<tag_type>(merged_tag_buffer)));
      thrust::merge(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                    old_pair_first,
                    old_pair_first + vertices_.size(),
                    new_pair_first,
                    new_pair_first + thrust::distance(vertex_first, vertex_last),
                    merged_pair_first);
      merged_vertices.resize(
        thrust::distance(
          merged_pair_first,
          thrust::unique(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                         merged_pair_first,
                         merged_pair_first + merged_vertices.size())),
        handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(
        merged_tag_buffer, merged_vertices.size(), handle_ptr_->get_stream());
      merged_vertices.shrink_to_fit(handle_ptr_->get_stream());
      shrink_to_fit_dataframe_buffer<tag_type>(merged_tag_buffer, handle_ptr_->get_stream());
      vertices_    = std::move(merged_vertices);
      tags_.buffer = std::move(merged_tag_buffer);
    } else {
      vertices_.resize(thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(tags_.buffer, vertices_.size(), handle_ptr_->get_stream());
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertex_first, tag_first));
      thrust::copy(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   pair_first,
                   pair_first + vertices_.size(),
                   thrust::make_zip_iterator(thrust::make_tuple(
                     vertices_.begin(), get_dataframe_buffer_begin<tag_type>(tags_.buffer))));
    }
  }

  /**
   * @ brief insert a list of vertices and vertx payloads to the bucket
   *
   * @param vertex_first Iterator pointing to the first (inclusive) element of the vertex list
   * stored in device memory.
   * @param vertex_last Iterator pointing to the last (exclusive) element of the vertex list stored
   * in device memory.
   * @param payload_first Iterator pointing to the first (inclusive) element of the vertex payloads
   * stored in device memory.
   * @param reduce_op Reduction operation to accumulate two payload_t variables, @reduce_op should
   * be a pure function with no side effect.
   */
  template <typename VertexIterator,
            typename PayloadIterator,
            typename ReduceOp,
            typename tag_type                                      = tag_t,
            typename payload_type                                  = payload_t,
            std::enable_if_t<std::is_same_v<tag_type, void> &&
                             !std::is_same_v<payload_type, void>>* = nullptr>
  void insert(VertexIterator vertex_first,
              VertexIterator vertex_last,
              PayloadIterator payload_first,
              ReduceOp reduce_op)
  {
    static_assert(
      std::is_same_v<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>);
    static_assert(
      std::is_same_v<typename std::iterator_traits<PayloadIterator>::value_type, payload_type>);

    if (vertices_.size() > 0) {
      rmm::device_uvector<vertex_t> merged_vertices(
        vertices_.size() + thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      auto merged_payload_buffer =
        allocate_dataframe_buffer<payload_type>(merged_vertices.size(), handle_ptr_->get_stream());
      thrust::merge_by_key(
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        vertices_.begin(),
        vertices_.end(),
        vertex_first,
        vertex_last,
        get_dataframe_buffer_begin<payload_type>(payloads_.buffer),
        payload_first,
        merged_vertices.begin(),
        get_dataframe_buffer_begin<payload_type>(merged_payload_buffer));

      // FIXME: if reduce_op is picking any, thrust::unique is sufficient (and this may be faster)
      vertices_.resize(merged_vertices.size(), handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(
        payloads_.buffer, vertices_.size(), handle_ptr_->get_stream());
      vertices_.resize(
        thrust::distance(
          vertices_.begin(),
          thrust::get<0>(thrust::reduce_by_key(
            rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
            merged_vertices.begin(),
            merged_vertices.end(),
            get_dataframe_buffer_begin<payload_type>(merged_payload_buffer),
            vertices_.begin(),
            get_dataframe_buffer_begin<payload_type>(payloads_.buffer),
            thrust::equal_to<vertex_t>(),
            reduce_op))),
        handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(
        payloads_.buffer, vertices_.size(), handle_ptr_->get_stream());
      vertices_.shrink_to_fit(handle_ptr_->get_stream());
      shrink_to_fit_dataframe_buffer<payload_type>(payloads_.buffer, handle_ptr_->get_stream());
    } else {
      vertices_.resize(thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(
        payloads_.buffer, vertices_.size(), handle_ptr_->get_stream());
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertex_first, payload_first));
      thrust::copy(
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        pair_first,
        pair_first + vertices_.size(),
        thrust::make_zip_iterator(thrust::make_tuple(
          vertices_.begin(), get_dataframe_buffer_begin<payload_type>(payloads_.buffer))));
    }
  }

  /**
   * @ brief insert a list of vertices, vertex tags, and vertx payloads to the bucket
   *
   * @param vertex_first Iterator pointing to the first (inclusive) element of the vertex list
   * stored in device memory.
   * @param vertex_last Iterator pointing to the last (exclusive) element of the vertex list stored
   * in device memory.
   * @param payload_first Iterator pointing to the first (inclusive) element of the vertex payloads
   * stored in device memory.
   * @param reduce_op Reduction operation to accumulate two payload_t variables, @reduce_op should
   * be a pure function with no side effect.
   */
  template <typename VertexIterator,
            typename TagIterator,
            typename PayloadIterator,
            typename ReduceOp,
            typename tag_type                                      = tag_t,
            typename payload_type                                  = payload_t,
            std::enable_if_t<std::is_same_v<tag_type, void> &&
                             !std::is_same_v<payload_type, void>>* = nullptr>
  void insert(VertexIterator vertex_first,
              VertexIterator vertex_last,
              TagIterator tag_first,
              PayloadIterator payload_first,
              ReduceOp reduce_op)
  {
    static_assert(
      std::is_same_v<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>);
    static_assert(std::is_same_v<typename std::iterator_traits<TagIterator>::value_type, tag_type>);
    static_assert(
      std::is_same_v<typename std::iterator_traits<PayloadIterator>::value_type, payload_type>);

    if (vertices_.size() > 0) {
      rmm::device_uvector<vertex_t> merged_vertices(
        vertices_.size() + thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      auto merged_tag_buffer =
        allocate_dataframe_buffer<tag_type>(merged_vertices.size(), handle_ptr_->get_stream());
      auto merged_payload_buffer =
        allocate_dataframe_buffer<payload_type>(merged_vertices.size(), handle_ptr_->get_stream());
      auto old_pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(vertices_.begin(), get_dataframe_buffer_begin<tag_type>(tags_.buffer)));
      auto new_pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertex_first, tag_first));
      auto merged_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        merged_vertices.begin(), get_dataframe_buffer_begin<tag_type>(merged_tag_buffer)));
      thrust::merge_by_key(
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        old_pair_first,
        old_pair_first + vertices_.size(),
        new_pair_first,
        new_pair_first + thrust::distance(vertex_first, vertex_last),
        get_dataframe_buffer_begin<payload_type>(payloads_.buffer),
        payload_first,
        merged_pair_first,
        get_dataframe_buffer_begin<payload_type>(merged_payload_buffer));

      // FIXME: if reduce_op is picking any, thrust::unique is sufficient (and this may be faster)
      vertices_.resize(merged_vertices.size(), handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(tags_.buffer, vertices_.size(), handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(
        payloads_.buffer, vertices_.size(), handle_ptr_->get_stream());
      auto reduced_pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(vertices_.begin(), get_dataframe_buffer_begin<tag_type>(tags_.buffer)));
      vertices_.resize(
        thrust::distance(
          reduced_pair_first,
          thrust::get<0>(thrust::reduce_by_key(
            rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
            merged_pair_first,
            merged_pair_first + merged_vertices.size(),
            get_dataframe_buffer_begin<payload_type>(merged_payload_buffer),
            reduced_pair_first,
            get_dataframe_buffer_begin<payload_type>(payloads_.buffer),
            thrust::equal_to<thrust::tuple<vertex_t, tag_type>>(),
            reduce_op))),
        handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(tags_.buffer, vertices_.size(), handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(
        payloads_.buffer, vertices_.size(), handle_ptr_->get_stream());
      vertices_.shrink_to_fit(handle_ptr_->get_stream());
      shrink_to_fit_dataframe_buffer<tag_type>(tags_.buffer, handle_ptr_->get_stream());
      shrink_to_fit_dataframe_buffer<payload_type>(payloads_.buffer, handle_ptr_->get_stream());
    } else {
      vertices_.resize(thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      resize_dataframe_buffer<tag_type>(tags_.buffer, vertices_.size(), handle_ptr_->get_stream());
      resize_dataframe_buffer<payload_type>(
        payloads_.buffer, vertices_.size(), handle_ptr_->get_stream());
      auto triplet_first =
        thrust::make_zip_iterator(thrust::make_tuple(vertex_first, tag_first, payload_first));
      thrust::copy(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   triplet_first,
                   triplet_first + vertices_.size(),
                   thrust::make_zip_iterator(thrust::make_tuple(
                     vertices_.begin(),
                     get_dataframe_buffer_begin<tag_type>(tags_.buffer),
                     get_dataframe_buffer_begin<payload_type>(payloads_.buffer))));
    }
  }

  size_t size() const { return vertices_.size(); }

  template <bool do_aggregate = is_multi_gpu>
  std::enable_if_t<do_aggregate, size_t> aggregate_size() const
  {
    return host_scalar_allreduce(
      handle_ptr_->get_comms(), vertices_.size(), handle_ptr_->get_stream());
  }

  template <bool do_aggregate = is_multi_gpu>
  std::enable_if_t<!do_aggregate, size_t> aggregate_size() const
  {
    return vertices_.size();
  }

  void resize(size_t size)
  {
    vertices_.resize(size, handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) {
      resize_dataframe_buffer<tag_t>(tags_.buffer, size, handle_ptr_->get_stream());
    }
  }

  void clear() { resize(0); }

  void shrink_to_fit()
  {
    vertices_.shrink_to_fit(handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) {
      shrink_to_fit_dataframe_buffer(tags_.buffer, handle_ptr_->get_stream());
    }
  }

  auto const begin() const
  {
    if constexpr (std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return vertices_.begin();
    } else if constexpr (!std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.begin(), get_dataframe_buffer_begin<tag_t>(tags_.buffer));
    } else if constexpr (std::is_same_v<tag_t, void> && !std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.begin(),
                             get_dataframe_buffer_begin<payload_t>(payloads_.buffer));
    } else {
      return std::make_tuple(vertices_.begin(),
                             get_dataframe_buffer_begin<tag_t>(tags_.buffer),
                             get_dataframe_buffer_begin<payload_t>(payloads_.buffer));
    }
  }

  auto begin()
  {
    if constexpr (std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return vertices_.begin();
    } else if constexpr (!std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.begin(), get_dataframe_buffer_begin<tag_t>(tags_.buffer));
    } else if constexpr (std::is_same_v<tag_t, void> && !std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.begin(),
                             get_dataframe_buffer_begin<payload_t>(payloads_.buffer));
    } else {
      return std::make_tuple(vertices_.begin(),
                             get_dataframe_buffer_begin<tag_t>(tags_.buffer),
                             get_dataframe_buffer_begin<payload_t>(payloads_.buffer));
    }
  }

  auto const end() const
  {
#if 1
    // auto first = begin();
    return begin() + vertices_.size();
#else
    if constexpr (std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return vertices_.end();
    } else if constexpr (!std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.end(), get_dataframe_buffer_end<tag_t>(tags_.buffer));
    } else if constexpr (std::is_same_v<tag_t, void> && !std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.end(),
                             get_dataframe_buffer_end<payload_t>(payloads_.buffer));
    } else {
      return std::make_tuple(vertices_.end(),
                             get_dataframe_buffer_end<tag_t>(tags_.buffer),
                             get_dataframe_buffer_end<payload_t>(payloads_.buffer));
    }
#endif
  }

  auto end()
  {
#if 1
    // auto first = begin();
    return begin() + vertices_.size();
#else
    if constexpr (std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return vertices_.end();
    } else if constexpr (!std::is_same_v<tag_t, void> && std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.end(), get_dataframe_buffer_end<tag_t>(tags_.buffer));
    } else if constexpr (std::is_same_v<tag_t, void> && !std::is_same_v<payload_t, void>) {
      return std::make_tuple(vertices_.end(),
                             get_dataframe_buffer_end<payload_t>(payloads_.buffer));
    } else {
      return std::make_tuple(vertices_.end(),
                             get_dataframe_buffer_end<tag_t>(tags_.buffer),
                             get_dataframe_buffer_end<payload_t>(payloads_.buffer));
    }
#endif
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_uvector<vertex_t> vertices_;
  detail::optional_buffer_t<tag_t> tags_;
  detail::optional_buffer_t<payload_t> payloads_;
};

template <typename vertex_t,
          typename tag_t     = void,
          typename payload_t = void,
          bool is_multi_gpu  = false,
          size_t num_buckets = 1>
class VertexFrontier {
  static_assert(std::is_same_v<tag_t, void> ||
                is_arithmetic_or_thrust_tuple_of_arithmetic<tag_t>::value);

 public:
  static size_t constexpr kNumBuckets = num_buckets;
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  VertexFrontier(raft::handle_t const& handle) : handle_ptr_(&handle)
  {
    for (size_t i = 0; i < num_buckets; ++i) { buckets_.emplace_back(handle); }
  }

  SortedUniqueElementBucket<vertex_t, tag_t, payload_t, is_multi_gpu>& get_bucket(size_t bucket_idx)
  {
    return buckets_[bucket_idx];
  }

  SortedUniqueElementBucket<vertex_t, tag_t, payload_t, is_multi_gpu> const& get_bucket(
    size_t bucket_idx) const
  {
    return buckets_[bucket_idx];
  }

  void swap_buckets(size_t bucket_idx0, size_t bucket_idx1)
  {
    std::swap(buckets_[bucket_idx0], buckets_[bucket_idx1]);
  }

  template <typename SplitOp,
            typename payload_type                                 = payload_t,
            std::enable_if_t<std::is_same_v<payload_type, void>>* = nullptr>
  void split_bucket(size_t this_bucket_idx,
                    std::vector<size_t> const& move_to_bucket_indices,
                    SplitOp split_op)
  {
    auto& this_bucket = get_bucket(this_bucket_idx);
    if (this_bucket.size() == 0) { return; }

    auto [new_this_bucket_size, insert_bucket_indices, insert_offsets, insert_sizes] =
      split_and_groupby_bucket(this_bucket_idx, move_to_bucket_indices, split_op);

    if constexpr (std::is_same_v<tag_t, void>) {
      for (size_t i = 0; i < insert_offsets.size(); ++i) {
        get_bucket(insert_bucket_indices[i])
          .insert(this_bucket.begin() + insert_offsets[i],
                  this_bucket.begin() + (insert_offsets[i] + insert_sizes[i]));
      }
    } else if constexpr (!std::is_same_v<tag_t, void>) {
      auto vertex_first = std::get<0>(this_bucket.begin());
      auto tag_first    = std::get<1>(this_bucket.begin());
      for (size_t i = 0; i < insert_offsets.size(); ++i) {
        get_bucket(insert_bucket_indices[i])
          .insert(vertex_first + insert_offsets[i],
                  vertex_first + (insert_offsets[i] + insert_sizes[i]),
                  tag_first + insert_offsets[i]);
      }
    }

    this_bucket.resize(new_this_bucket_size);
    this_bucket.shrink_to_fit();
  }

  template <typename SplitOp,
            typename ReduceOp,
            typename payload_type                                  = payload_t,
            std::enable_if_t<!std::is_same_v<payload_type, void>>* = nullptr>
  void split_bucket(size_t this_bucket_idx,
                    std::vector<size_t> const& move_to_bucket_indices,
                    SplitOp split_op,
                    ReduceOp reduce_op)
  {
    auto& this_bucket = get_bucket(this_bucket_idx);
    if (this_bucket.size() == 0) { return; }

    auto [new_this_bucket_size, insert_bucket_indices, insert_offsets, insert_sizes] =
      split_and_groupby_bucket(this_bucket_idx, move_to_bucket_indices, split_op);

    if constexpr (std::is_same_v<tag_t, void>) {
      auto vertex_first  = std::get<0>(this_bucket.begin());
      auto payload_first = std::get<1>(this_bucket.begin());
      for (size_t i = 0; i < insert_offsets.size(); ++i) {
        get_bucket(insert_bucket_indices[i])
          .insert(vertex_first + insert_offsets[i],
                  vertex_first + (insert_offsets[i] + insert_sizes[i]),
                  payload_first + insert_offsets[i],
                  reduce_op);
      }
    } else {
      auto vertex_first  = std::get<0>(this_bucket.begin());
      auto tag_first     = std::get<1>(this_bucket.begin());
      auto payload_first = std::get<2>(this_bucket.begin());
      for (size_t i = 0; i < insert_offsets.size(); ++i) {
        get_bucket(insert_bucket_indices[i])
          .insert(vertex_first + insert_offsets[i],
                  vertex_first + (insert_offsets[i] + insert_sizes[i]),
                  tag_first + insert_offsets[i],
                  payload_first + insert_offsets[i],
                  reduce_op);
      }
    }

    this_bucket.resize(new_this_bucket_size);
    this_bucket.shrink_to_fit();
  }

  // FIXME: this function is not part of the public stable API. This function should be private, but
  // nvcc currently disallows the enclosing parent function for an extended __device__ lambda to
  // have private or protected access
  template <typename SplitOp>
  std::tuple<size_t, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
  split_and_groupby_bucket(size_t this_bucket_idx,
                           std::vector<size_t> const& move_to_bucket_indices,
                           SplitOp split_op)
  {
    auto& this_bucket = get_bucket(this_bucket_idx);

    // 1. apply split_op to each bucket element

    static_assert(kNumBuckets <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(this_bucket.size(), handle_ptr_->get_stream());
    thrust::transform(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                      this_bucket.begin(),
                      this_bucket.end(),
                      bucket_indices.begin(),
                      [split_op] __device__(auto v) { return static_cast<uint8_t>(split_op(v)); });

    // 2. remove elements with the invalid bucket indices

    auto pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), this_bucket.begin()));
    this_bucket.resize(thrust::distance(
      pair_first,
      thrust::remove_if(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                        pair_first,
                        pair_first + bucket_indices.size(),
                        [invalid_bucket_idx = static_cast<uint8_t>(kInvalidBucketIdx)] __device__(
                          auto pair) { return thrust::get<0>(pair) == invalid_bucket_idx; })));
    bucket_indices.resize(this_bucket.size(), handle_ptr_->get_stream());
    this_bucket.shrink_to_fit();
    bucket_indices.shrink_to_fit(handle_ptr_->get_stream_view());

    // 3. separte the elements to stay in this bucket from the elements to be moved to other buckets

    pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), this_bucket.begin()));
    auto new_this_bucket_size = static_cast<size_t>(thrust::distance(
      pair_first,
      thrust::stable_partition(  // stalbe_partition to maintain sorted order within each bucket
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        pair_first,
        pair_first + bucket_indices.size(),
        [this_bucket_idx = static_cast<uint8_t>(this_bucket_idx)] __device__(auto pair) {
          return thrust::get<0>(pair) == this_bucket_idx;
        })));

    // 4. group the remaining elements by their target bucket indices

    std::vector<size_t> insert_bucket_indices{};
    std::vector<size_t> insert_offsets{};
    std::vector<size_t> insert_sizes{};
    if (move_to_bucket_indices.size() == 1) {
      insert_bucket_indices = move_to_bucket_indices;
      insert_offsets        = {new_this_bucket_size};
      insert_sizes          = {static_cast<size_t>(
        thrust::distance(this_bucket.begin() + new_this_bucket_size, this_bucket.end()))};
    } else if (move_to_bucket_indices.size() == 2) {
      auto next_bucket_size = static_cast<size_t>(thrust::distance(
        pair_first + new_this_bucket_size,
        thrust::stable_partition(  // stalbe_partition to maintain sorted order within each bucket
          rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
          pair_first + new_this_bucket_size,
          pair_first + bucket_indices.size(),
          [next_bucket_idx = static_cast<uint8_t>(move_to_bucket_indices[0])] __device__(
            auto pair) { return thrust::get<0>(pair) == next_bucket_idx; })));
      insert_bucket_indices = move_to_bucket_indices;
      insert_offsets        = {new_this_bucket_size, new_this_bucket_size + next_bucket_size};
      insert_sizes          = {
        next_bucket_size,
        static_cast<size_t>(thrust::distance(
          this_bucket.begin() + (new_this_bucket_size + next_bucket_size), this_bucket.end()))};
    } else {
      thrust::stable_sort(  // stalbe_sort to maintain sorted order within each bucket
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        pair_first + new_this_bucket_size,
        pair_first + bucket_indices.size(),
        [] __device__(auto lhs, auto rhs) { return thrust::get<0>(lhs) < thrust::get<1>(rhs); });
      rmm::device_uvector<uint8_t> d_indices(move_to_bucket_indices.size(),
                                             handle_ptr_->get_stream());
      rmm::device_uvector<size_t> d_counts(d_indices.size(), handle_ptr_->get_stream());
      auto it = thrust::reduce_by_key(
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        bucket_indices.begin() + new_this_bucket_size,
        bucket_indices.end(),
        thrust::make_constant_iterator(size_t{1}),
        d_indices.begin(),
        d_counts.begin());
      d_indices.resize(thrust::distance(d_indices.begin(), thrust::get<0>(it)),
                       handle_ptr_->get_stream());
      d_counts.resize(d_indices.size(), handle_ptr_->get_stream());
      std::vector<uint8_t> h_indices(d_indices.size());
      std::vector<size_t> h_counts(h_indices.size());
      raft::update_host(
        h_indices.data(), d_indices.data(), d_indices.size(), handle_ptr_->get_stream());
      raft::update_host(
        h_counts.data(), d_counts.data(), d_counts.size(), handle_ptr_->get_stream());
      handle_ptr_->get_stream_view().synchronize();

      auto offset = new_this_bucket_size;
      for (size_t i = 0; i < h_indices.size(); ++i) {
        insert_bucket_indices[i] = static_cast<size_t>(h_indices[i]);
        insert_offsets[i]        = offset;
        insert_sizes[i]          = h_counts[i];
        offset += insert_sizes[i];
      }
    }

    return std::make_tuple(
      new_this_bucket_size, insert_bucket_indices, insert_offsets, insert_sizes);
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  std::vector<SortedUniqueElementBucket<vertex_t, tag_t, payload_t, is_multi_gpu>> buckets_{};
};

}  // namespace experimental
}  // namespace cugraph

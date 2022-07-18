/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/thrust_tuple_utils.cuh>

#include <raft/handle.hpp>
#include <raft/span.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <optional>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename ValueIterator>
class edge_partition_major_property_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  edge_partition_major_property_device_view_t() = default;

  edge_partition_major_property_device_view_t(
    ValueIterator value_first)  // for single-GPU only and for advanced users
    : value_first_(value_first)
  {
    set_local_edge_partition_idx(size_t{0});
  }

  edge_partition_major_property_device_view_t(
    raft::host_span<vertex_t const> edge_partition_major_value_start_offsets,
    raft::host_span<vertex_t const> edge_partition_major_range_firsts,
    ValueIterator value_first)
    : edge_partition_major_value_start_offsets_(edge_partition_major_value_start_offsets),
      edge_partition_major_range_firsts_(edge_partition_major_range_firsts),
      value_first_(value_first)
  {
    set_local_edge_partition_idx(size_t{0});
  }

  edge_partition_major_property_device_view_t(
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_keys,
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_key_chunk_start_offsets,
    size_t key_chunk_size,
    raft::host_span<vertex_t const> edge_partition_major_value_start_offsets,
    raft::host_span<vertex_t const> edge_partition_major_range_firsts,
    ValueIterator value_first)
    : edge_partition_keys_(edge_partition_keys),
      edge_partition_key_chunk_start_offsets_(edge_partition_key_chunk_start_offsets),
      key_chunk_size_(key_chunk_size),
      edge_partition_major_value_start_offsets_(edge_partition_major_value_start_offsets),
      edge_partition_major_range_firsts_(edge_partition_major_range_firsts),
      value_first_(value_first)
  {
    set_local_edge_partition_idx(size_t{0});
  }

  void set_local_edge_partition_idx(size_t partition_idx)
  {
    if (edge_partition_keys_) {
      this_edge_partition_keys_ = (*edge_partition_keys_)[partition_idx];
      this_edge_partition_key_chunk_start_offsets_ =
        (*edge_partition_key_chunk_start_offsets_)[partition_idx];
    }

    assert((partition_idx == size_t{0}) || edge_partition_major_value_start_offsets_);
    assert((partition_idx == size_t{0}) || edge_partition_major_range_firsts_);
    this_edge_partition_value_first_ =
      value_first_ + (edge_partition_major_value_start_offsets_
                        ? (*edge_partition_major_value_start_offsets_)[partition_idx]
                        : vertex_t{0});
    this_edge_partition_major_range_first_ =
      edge_partition_major_range_firsts_ ? (*edge_partition_major_range_firsts_)[partition_idx]
                                         : vertex_t{0};
  }

  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> keys() const
  {
    return edge_partition_keys_
             ? std::optional<
                 raft::host_span<raft::device_span<vertex_t const> const>>{*edge_partition_keys_}
             : std::nullopt;
  }

  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> key_chunk_start_offsets()
    const
  {
    return edge_partition_key_chunk_start_offsets_
             ? std::optional<raft::host_span<
                 raft::device_span<vertex_t const> const>>{*edge_partition_key_chunk_start_offsets_}
             : std::nullopt;
  }

  std::optional<size_t> key_chunk_size() const
  {
    return key_chunk_size_ ? std::optional<size_t>{*key_chunk_size_} : std::nullopt;
  }

  ValueIterator value_first() const { return value_first_; }

  std::optional<raft::host_span<vertex_t const>> edge_partition_major_value_start_offsets() const
  {
    return edge_partition_major_value_start_offsets_
             ? std::optional<
                 raft::host_span<vertex_t const>>{*edge_partition_major_value_start_offsets_}
             : std::nullopt;
  }

  std::optional<raft::host_span<vertex_t const>> edge_partition_major_range_firsts() const
  {
    return edge_partition_major_range_firsts_
             ? std::optional<raft::host_span<vertex_t const>>{*edge_partition_major_range_firsts_}
             : std::nullopt;
  }

  __device__ ValueIterator get_iter(vertex_t offset) const
  {
    auto value_offset = offset;
    if (this_edge_partition_keys_) {
      auto chunk_idx = static_cast<size_t>(offset) / (*key_chunk_size_);
      auto it =
        thrust::lower_bound(thrust::seq,
                            (*this_edge_partition_keys_).begin() +
                              (*this_edge_partition_key_chunk_start_offsets_)[chunk_idx],
                            (*this_edge_partition_keys_).begin() +
                              (*this_edge_partition_key_chunk_start_offsets_)[chunk_idx + 1],
                            this_edge_partition_major_range_first_ + offset);
      assert((it != (*this_edge_partition_keys_).begin() +
                      (*this_edge_partition_key_chunk_start_offsets_)[chunk_idx + 1]) &&
             (*it == (this_edge_partition_major_range_first_ + offset)));
      value_offset = (*this_edge_partition_key_chunk_start_offsets_)[chunk_idx] +
                     static_cast<vertex_t>(thrust::distance(
                       (*this_edge_partition_keys_).begin() +
                         (*this_edge_partition_key_chunk_start_offsets_)[chunk_idx],
                       it));
    }
    return this_edge_partition_value_first_ + value_offset;
  }

  __device__ value_type get(vertex_t offset) const { return *get_iter(offset); }

 private:
  thrust::optional<raft::host_span<raft::device_span<vertex_t const> const>> edge_partition_keys_{
    thrust::nullopt};
  thrust::optional<raft::host_span<raft::device_span<vertex_t const> const>>
    edge_partition_key_chunk_start_offsets_{thrust::nullopt};
  thrust::optional<size_t> key_chunk_size_{thrust::nullopt};

  thrust::optional<raft::host_span<vertex_t const>> edge_partition_major_value_start_offsets_{
    thrust::nullopt};
  thrust::optional<raft::host_span<vertex_t const>> edge_partition_major_range_firsts_{
    thrust::nullopt};
  ValueIterator value_first_{};

  thrust::optional<raft::device_span<vertex_t const>> this_edge_partition_keys_{thrust::nullopt};
  thrust::optional<raft::device_span<vertex_t const>> this_edge_partition_key_chunk_start_offsets_{
    thrust::nullopt};
  ValueIterator this_edge_partition_value_first_{};
  vertex_t this_edge_partition_major_range_first_{};
};

template <typename vertex_t, typename ValueIterator>
class edge_partition_minor_property_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  edge_partition_minor_property_device_view_t() = default;

  edge_partition_minor_property_device_view_t(ValueIterator value_first, vertex_t minor_range_first)
    : value_first_(value_first), minor_range_first_(minor_range_first)
  {
  }

  edge_partition_minor_property_device_view_t(
    raft::device_span<vertex_t const> keys,
    raft::device_span<vertex_t const> key_chunk_start_offsets,
    size_t key_chunk_size,
    ValueIterator value_first,
    vertex_t minor_range_first)
    : keys_(keys),
      key_chunk_start_offsets_(key_chunk_start_offsets),
      key_chunk_size_(key_chunk_size),
      value_first_(value_first),
      minor_range_first_(minor_range_first)
  {
  }

  std::optional<raft::device_span<vertex_t const>> keys() const
  {
    return keys_ ? std::optional<raft::device_span<vertex_t const>>{*keys_} : std::nullopt;
  }

  ValueIterator value_first() const { return value_first_; }

  __device__ ValueIterator get_iter(vertex_t offset) const
  {
    auto value_offset = offset;
    if (keys_) {
      auto chunk_idx = static_cast<size_t>(offset) / (*key_chunk_size_);
      auto it        = thrust::lower_bound(thrust::seq,
                                    (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx],
                                    (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1],
                                    minor_range_first_ + offset);
      assert((it != (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1]) &&
             (*it == (minor_range_first_ + offset)));
      value_offset = (*key_chunk_start_offsets_)[chunk_idx] +
                     static_cast<vertex_t>(thrust::distance(
                       (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx], it));
    }
    return value_first_ + value_offset;
  }

  __device__ value_type get(vertex_t offset) const { return *get_iter(offset); }

 private:
  thrust::optional<raft::device_span<vertex_t const>> keys_{thrust::nullopt};
  thrust::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{thrust::nullopt};
  thrust::optional<size_t> key_chunk_size_{thrust::nullopt};

  ValueIterator value_first_{};
  vertex_t minor_range_first_{};
};

template <typename vertex_t, typename T>
class edge_partition_major_property_t {
 public:
  edge_partition_major_property_t(raft::handle_t const& handle)
    : buffer_(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))
  {
  }

  edge_partition_major_property_t(raft::handle_t const& handle,
                                  std::vector<vertex_t>&& edge_partition_major_value_start_offsets,
                                  std::vector<vertex_t>&& edge_partition_major_range_firsts)
    : buffer_(allocate_dataframe_buffer<T>(edge_partition_major_value_start_offsets.back(),
                                           handle.get_stream())),
      edge_partition_major_value_start_offsets_(
        std::move(edge_partition_major_value_start_offsets)),
      edge_partition_major_range_firsts_(std::move(edge_partition_major_range_firsts))
  {
  }

  edge_partition_major_property_t(
    raft::handle_t const& handle,
    raft::host_span<raft::device_span<vertex_t const> const> keys,
    raft::host_span<raft::device_span<vertex_t const> const> key_chunk_start_offsets,
    size_t key_chunk_size,
    std::vector<vertex_t>&& edge_partition_major_value_start_offsets,
    std::vector<vertex_t>&& edge_partition_major_range_firsts)
    : edge_partition_keys_(keys),
      edge_partition_key_chunk_start_offsets_(key_chunk_start_offsets),
      key_chunk_size_(key_chunk_size),
      buffer_(allocate_dataframe_buffer<T>(edge_partition_major_value_start_offsets.back(),
                                           handle.get_stream())),
      edge_partition_major_value_start_offsets_(
        std::move(edge_partition_major_value_start_offsets)),
      edge_partition_major_range_firsts_(std::move(edge_partition_major_range_firsts))
  {
  }

  void clear(raft::handle_t const& handle)
  {
    edge_partition_keys_                    = std::nullopt;
    edge_partition_key_chunk_start_offsets_ = std::nullopt;
    key_chunk_size_                         = std::nullopt;

    resize_dataframe_buffer(buffer_, size_t{0}, handle.get_stream());
    shrink_to_fit_dataframe_buffer(buffer_, handle.get_stream());

    edge_partition_major_value_start_offsets_.clear();
    edge_partition_major_value_start_offsets_.shrink_to_fit();
    edge_partition_major_range_firsts_.clear();
    edge_partition_major_range_firsts_.shrink_to_fit();
  }

  void fill(raft::handle_t const& handle, T value)
  {
    thrust::fill(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(buffer_),
                 get_dataframe_buffer_end(buffer_),
                 value);
  }

  auto keys(size_t partition_idx)
  {
    return edge_partition_keys_ ? std::optional<raft::device_span<vertex_t const>>{(
                                    *edge_partition_keys_)[partition_idx]}
                                : std::nullopt;
  }

  auto value_first() { return get_dataframe_buffer_begin(buffer_); }

  auto device_view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);

    if (edge_partition_keys_) {
      return edge_partition_major_property_device_view_t<vertex_t, decltype(value_first)>(
        *edge_partition_keys_,
        *edge_partition_key_chunk_start_offsets_,
        *key_chunk_size_,
        raft::host_span<vertex_t const>(edge_partition_major_value_start_offsets_.data(),
                                        edge_partition_major_value_start_offsets_.size()),
        raft::host_span<vertex_t const>(edge_partition_major_range_firsts_.data(),
                                        edge_partition_major_range_firsts_.size()),
        value_first);
    } else {
      return edge_partition_major_property_device_view_t<vertex_t, decltype(value_first)>(
        raft::host_span<vertex_t const>(edge_partition_major_value_start_offsets_.data(),
                                        edge_partition_major_value_start_offsets_.size()),
        raft::host_span<vertex_t const>(edge_partition_major_range_firsts_.data(),
                                        edge_partition_major_range_firsts_.size()),
        value_first);
    }
  }

  auto mutable_device_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);

    if (edge_partition_keys_) {
      return edge_partition_major_property_device_view_t<vertex_t, decltype(value_first)>(
        *edge_partition_keys_,
        *edge_partition_key_chunk_start_offsets_,
        *key_chunk_size_,
        raft::host_span<vertex_t const>(edge_partition_major_value_start_offsets_.data(),
                                        edge_partition_major_value_start_offsets_.size()),
        raft::host_span<vertex_t const>(edge_partition_major_range_firsts_.data(),
                                        edge_partition_major_range_firsts_.size()),
        value_first);
    } else {
      return edge_partition_major_property_device_view_t<vertex_t, decltype(value_first)>(
        raft::host_span<vertex_t const>(edge_partition_major_value_start_offsets_.data(),
                                        edge_partition_major_value_start_offsets_.size()),
        raft::host_span<vertex_t const>(edge_partition_major_range_firsts_.data(),
                                        edge_partition_major_range_firsts_.size()),
        value_first);
    }
  }

 private:
  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> edge_partition_keys_{
    std::nullopt};
  std::optional<raft::host_span<raft::device_span<vertex_t const> const>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{})) buffer_;
  std::vector<vertex_t> edge_partition_major_value_start_offsets_{};
  std::vector<vertex_t> edge_partition_major_range_firsts_{};
};

template <typename vertex_t, typename T>
class edge_partition_minor_property_t {
 public:
  edge_partition_minor_property_t(raft::handle_t const& handle)
    : buffer_(allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream())),
      minor_range_first_(vertex_t{0})
  {
  }

  edge_partition_minor_property_t(raft::handle_t const& handle,
                                  vertex_t buffer_size,
                                  vertex_t minor_range_first)
    : buffer_(allocate_dataframe_buffer<T>(buffer_size, handle.get_stream())),
      minor_range_first_(minor_range_first)
  {
  }

  edge_partition_minor_property_t(raft::handle_t const& handle,
                                  raft::device_span<vertex_t const> keys,
                                  raft::device_span<vertex_t const> key_chunk_start_offsets,
                                  size_t key_chunk_size,
                                  vertex_t minor_range_first)
    : keys_(keys),
      key_chunk_start_offsets_(key_chunk_start_offsets),
      key_chunk_size_(key_chunk_size),
      buffer_(allocate_dataframe_buffer<T>(keys.size(), handle.get_stream())),
      minor_range_first_(minor_range_first)
  {
  }

  void clear(raft::handle_t const& handle)
  {
    keys_                    = std::nullopt;
    key_chunk_start_offsets_ = std::nullopt;
    key_chunk_size_          = std::nullopt;

    resize_dataframe_buffer(buffer_, size_t{0}, handle.get_stream());
    shrink_to_fit_dataframe_buffer(buffer_, handle.get_stream());
    minor_range_first_ = vertex_t{0};
  }

  void fill(raft::handle_t const& handle, T value)
  {
    thrust::fill(handle.get_thrust_policy(),
                 value_first(),
                 value_first() + size_dataframe_buffer(buffer_),
                 value);
  }

  auto keys() { return keys_; }

  auto value_first() { return get_dataframe_buffer_begin(buffer_); }

  auto device_view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);
    if (keys_) {
      return edge_partition_minor_property_device_view_t<vertex_t, decltype(value_first)>(
        *keys_, *key_chunk_start_offsets_, *key_chunk_size_, value_first, minor_range_first_);
    } else {
      return edge_partition_minor_property_device_view_t<vertex_t, decltype(value_first)>(
        value_first, minor_range_first_);
    }
  }

  auto mutable_device_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);
    if (keys_) {
      return edge_partition_minor_property_device_view_t<vertex_t, decltype(value_first)>(
        *keys_, *key_chunk_start_offsets_, *key_chunk_size_, value_first, minor_range_first_);
    } else {
      return edge_partition_minor_property_device_view_t<vertex_t, decltype(value_first)>(
        value_first, minor_range_first_);
    }
  }

 private:
  std::optional<raft::device_span<vertex_t const>> keys_{std::nullopt};
  std::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{})) buffer_;
  vertex_t minor_range_first_{};
};

template <typename Iterator,
          typename std::enable_if_t<std::is_arithmetic<
            typename std::iterator_traits<Iterator>::value_type>::value>* = nullptr>
auto to_thrust_tuple(Iterator iter)
{
  return thrust::make_tuple(iter);
}

template <typename Iterator,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<
            typename std::iterator_traits<Iterator>::value_type>::value>* = nullptr>
auto to_thrust_tuple(Iterator iter)
{
  return iter.get_iterator_tuple();
}

template <typename T, typename... Ts>
decltype(auto) get_first_of_pack(T&& t, Ts&&...)
{
  return std::forward<T>(t);
}

}  // namespace detail

template <typename GraphViewType, typename T>
class edge_partition_src_property_t {
 public:
  using value_type = T;
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  edge_partition_src_property_t(raft::handle_t const& handle) : property_(handle) {}

  edge_partition_src_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
    : property_(handle)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    auto key_chunk_size = graph_view.local_sorted_unique_edge_src_chunk_size();
    if (key_chunk_size) {
      if constexpr (GraphViewType::is_multi_gpu) {
        if constexpr (GraphViewType::is_storage_transposed) {
          property_ = detail::edge_partition_minor_property_t<vertex_t, T>(
            handle,
            *(graph_view.local_sorted_unique_edge_srcs()),
            *(graph_view.local_sorted_unique_edge_src_chunk_start_offsets()),
            *key_chunk_size,
            graph_view.local_edge_partition_src_range_first());
        } else {
          edge_partition_keys_ = std::vector<raft::device_span<vertex_t const>>(
            graph_view.number_of_local_edge_partitions());
          edge_partition_key_chunk_start_offsets_ = std::vector<raft::device_span<vertex_t const>>(
            graph_view.number_of_local_edge_partitions());
          std::vector<vertex_t> major_value_start_offsets(
            graph_view.number_of_local_edge_partitions() + 1, vertex_t{0});
          std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
          for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
            (*edge_partition_keys_)[i] = *(graph_view.local_sorted_unique_edge_srcs(i));
            (*edge_partition_key_chunk_start_offsets_)[i] =
              *(graph_view.local_sorted_unique_edge_src_chunk_start_offsets(i));
            major_value_start_offsets[i + 1] =
              major_value_start_offsets[i] + (*edge_partition_keys_)[i].size();
            major_range_firsts[i] = graph_view.local_edge_partition_src_range_first(i);
          }
          property_ = detail::edge_partition_major_property_t<vertex_t, T>(
            handle,
            raft::host_span<raft::device_span<vertex_t const>>((*edge_partition_keys_).data(),
                                                               (*edge_partition_keys_).size()),
            raft::host_span<raft::device_span<vertex_t const>>(
              (*edge_partition_key_chunk_start_offsets_).data(),
              (*edge_partition_key_chunk_start_offsets_).size()),
            *key_chunk_size,
            std::move(major_value_start_offsets),
            std::move(major_range_firsts));
        }
      } else {
        assert(false);
      }
    } else {
      if constexpr (GraphViewType::is_storage_transposed) {
        property_ = detail::edge_partition_minor_property_t<vertex_t, T>(
          handle,
          graph_view.local_edge_partition_src_range_size(),
          graph_view.local_edge_partition_src_range_first());
      } else {
        std::vector<vertex_t> major_value_start_offsets(
          graph_view.number_of_local_edge_partitions() + 1, vertex_t{0});
        std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          major_value_start_offsets[i + 1] =
            major_value_start_offsets[i] + graph_view.local_edge_partition_src_range_size(i);
          major_range_firsts[i] = graph_view.local_edge_partition_src_range_first(i);
        }
        property_ = detail::edge_partition_major_property_t<vertex_t, T>(
          handle, std::move(major_value_start_offsets), std::move(major_range_firsts));
      }
    }
  }

  void clear(raft::handle_t const& handle)
  {
    property_.clear(handle);

    edge_partition_keys_                    = std::nullopt;
    edge_partition_key_chunk_start_offsets_ = std::nullopt;
  }

  void fill(raft::handle_t const& handle, T value) { property_.fill(handle, value); }

  template <bool transposed = GraphViewType::is_storage_transposed>
  std::enable_if_t<transposed,
                   std::optional<raft::device_span<typename GraphViewType::vertex_type const>>>
  keys()
  {
    return property_.keys();
  }

  template <bool transposed = GraphViewType::is_storage_transposed>
  std::enable_if_t<!transposed,
                   std::optional<raft::device_span<typename GraphViewType::vertex_type const>>>
  keys(size_t partition_idx)
  {
    return property_.keys(partition_idx);
  }

  auto value_first() { return property_.value_first(); }

  auto device_view() const { return property_.device_view(); }
  auto mutable_device_view() { return property_.mutable_device_view(); }

 private:
  std::conditional_t<
    GraphViewType::is_storage_transposed,
    detail::edge_partition_minor_property_t<typename GraphViewType::vertex_type, T>,
    detail::edge_partition_major_property_t<typename GraphViewType::vertex_type, T>>
    property_;

  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_keys_{std::nullopt};
  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
};

template <typename GraphViewType, typename T>
class edge_partition_dst_property_t {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  edge_partition_dst_property_t(raft::handle_t const& handle) : property_(handle) {}

  edge_partition_dst_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
    : property_(handle)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    auto key_chunk_size = graph_view.local_sorted_unique_edge_dst_chunk_size();
    if (key_chunk_size) {
      if constexpr (GraphViewType::is_multi_gpu) {
        if constexpr (GraphViewType::is_storage_transposed) {
          edge_partition_keys_ = std::vector<raft::device_span<vertex_t const>>(
            graph_view.number_of_local_edge_partitions());
          edge_partition_key_chunk_start_offsets_ = std::vector<raft::device_span<vertex_t const>>(
            graph_view.number_of_local_edge_partitions());
          std::vector<vertex_t> major_value_start_offsets(
            graph_view.number_of_local_edge_partitions() + 1, vertex_t{0});
          std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
          for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
            (*edge_partition_keys_)[i] = *(graph_view.local_sorted_unique_edge_dsts(i));
            (*edge_partition_key_chunk_start_offsets_)[i] =
              *(graph_view.local_sorted_unique_edge_dst_chunk_start_offsets(i));
            major_value_start_offsets[i + 1] =
              major_value_start_offsets[i] + (*edge_partition_keys_)[i].size();
            major_range_firsts[i] = graph_view.local_edge_partition_dst_range_first(i);
          }
          property_ = detail::edge_partition_major_property_t<vertex_t, T>(
            handle,
            raft::host_span<raft::device_span<vertex_t const>>((*edge_partition_keys_).data(),
                                                               (*edge_partition_keys_).size()),
            raft::host_span<raft::device_span<vertex_t const>>(
              (*edge_partition_key_chunk_start_offsets_).data(),
              (*edge_partition_key_chunk_start_offsets_).size()),
            *key_chunk_size,
            std::move(major_value_start_offsets),
            std::move(major_range_firsts));
        } else {
          property_ = detail::edge_partition_minor_property_t<vertex_t, T>(
            handle,
            *(graph_view.local_sorted_unique_edge_dsts()),
            *(graph_view.local_sorted_unique_edge_dst_chunk_start_offsets()),
            *key_chunk_size,
            graph_view.local_edge_partition_dst_range_first());
        }
      } else {
        assert(false);
      }
    } else {
      if constexpr (GraphViewType::is_storage_transposed) {
        std::vector<vertex_t> major_value_start_offsets(
          graph_view.number_of_local_edge_partitions() + 1, vertex_t{0});
        std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          major_value_start_offsets[i + 1] =
            major_value_start_offsets[i] + graph_view.local_edge_partition_dst_range_size(i);
          major_range_firsts[i] = graph_view.local_edge_partition_dst_range_first(i);
        }
        property_ = detail::edge_partition_major_property_t<vertex_t, T>(
          handle, std::move(major_value_start_offsets), std::move(major_range_firsts));
      } else {
        property_ = detail::edge_partition_minor_property_t<vertex_t, T>(
          handle,
          graph_view.local_edge_partition_dst_range_size(),
          graph_view.local_edge_partition_dst_range_first());
      }
    }
  }

  void clear(raft::handle_t const& handle)
  {
    property_.clear(handle);

    edge_partition_keys_                    = std::nullopt;
    edge_partition_key_chunk_start_offsets_ = std::nullopt;
  }

  void fill(raft::handle_t const& handle, T value) { property_.fill(handle, value); }

  template <bool transposed = GraphViewType::is_storage_transposed>
  std::enable_if_t<!transposed,
                   std::optional<raft::device_span<typename GraphViewType::vertex_type const>>>
  keys()
  {
    return property_.keys();
  }

  template <bool transposed = GraphViewType::is_storage_transposed>
  std::enable_if_t<transposed,
                   std::optional<raft::device_span<typename GraphViewType::vertex_type const>>>
  keys(size_t partition_idx)
  {
    return property_.keys(partition_idx);
  }

  auto value_first() { return property_.value_first(); }

  auto device_view() const { return property_.device_view(); }
  auto mutable_device_view() { return property_.mutable_device_view(); }

 private:
  std::conditional_t<
    GraphViewType::is_storage_transposed,
    detail::edge_partition_major_property_t<typename GraphViewType::vertex_type, T>,
    detail::edge_partition_minor_property_t<typename GraphViewType::vertex_type, T>>
    property_;

  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_keys_{std::nullopt};
  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
};

template <typename vertex_t>
class dummy_property_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  void set_local_edge_partition_idx(size_t partition_idx) {}  // no-op

  __device__ auto get(vertex_t offset) const { return thrust::nullopt; }
};

template <typename vertex_t>
class dummy_property_t {
 public:
  using value_type = thrust::nullopt_t;

  auto device_view() const { return dummy_property_device_view_t<vertex_t>{}; }
};

template <typename vertex_t, typename... Ts>
auto device_view_concat(
  detail::edge_partition_major_property_device_view_t<vertex_t, Ts> const&... device_views)
{
  auto concat_first = thrust::make_zip_iterator(
    thrust_tuple_cat(detail::to_thrust_tuple(device_views.value_first())...));
  auto first = detail::get_first_of_pack(device_views...);
  if (first.key_chunk_size()) {
    return detail::edge_partition_major_property_device_view_t<vertex_t, decltype(concat_first)>(
      *(first.keys()),
      *(first.key_chunk_start_offsets()),
      *(first.key_chunk_size()),
      *(first.edge_partition_major_value_start_offsets()),
      *(first.edge_partition_major_range_firsts()),
      concat_first);
  } else if (first.edge_partition_major_value_start_offsets()) {
    return detail::edge_partition_major_property_device_view_t<vertex_t, decltype(concat_first)>(
      *(first.edge_partition_major_value_start_offsets()),
      *(first.edge_partition_major_range_firsts()),
      concat_first);
  } else {
    return detail::edge_partition_major_property_device_view_t<vertex_t, decltype(concat_first)>(
      concat_first);
  }
}

}  // namespace cugraph

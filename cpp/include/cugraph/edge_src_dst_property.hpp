/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <optional>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename ValueIterator>
class edge_major_property_view_t {
 public:
  using value_type     = typename thrust::iterator_traits<ValueIterator>::value_type;
  using value_iterator = ValueIterator;

  edge_major_property_view_t() = default;

  edge_major_property_view_t(
    ValueIterator value_first)  // for single-GPU only and for advanced users
    : edge_partition_value_firsts_(std::vector<ValueIterator>{value_first}),
      edge_partition_major_range_firsts_(std::vector<vertex_t>{vertex_t{0}})
  {
  }

  edge_major_property_view_t(std::vector<ValueIterator> const& edge_partition_value_firsts,
                             std::vector<vertex_t> const& edge_partition_major_range_firsts)
    : edge_partition_value_firsts_(edge_partition_value_firsts),
      edge_partition_major_range_firsts_(edge_partition_major_range_firsts)
  {
  }

  edge_major_property_view_t(
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_keys,
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_key_chunk_start_offsets,
    size_t key_chunk_size,
    std::vector<ValueIterator> const& edge_partition_value_firsts,
    std::vector<vertex_t> const& edge_partition_major_range_firsts)
    : edge_partition_keys_(edge_partition_keys),
      edge_partition_key_chunk_start_offsets_(edge_partition_key_chunk_start_offsets),
      key_chunk_size_(key_chunk_size),
      edge_partition_value_firsts_(edge_partition_value_firsts),
      edge_partition_major_range_firsts_(edge_partition_major_range_firsts)
  {
  }

  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> keys() const
  {
    return edge_partition_keys_;
  }

  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> key_chunk_start_offsets()
    const
  {
    return edge_partition_key_chunk_start_offsets_;
  }

  std::optional<size_t> key_chunk_size() const { return key_chunk_size_; }

  std::vector<ValueIterator> const& value_firsts() const { return edge_partition_value_firsts_; }

  std::vector<vertex_t> const& major_range_firsts() const
  {
    return edge_partition_major_range_firsts_;
  }

 private:
  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> edge_partition_keys_{
    std::nullopt};
  std::optional<raft::host_span<raft::device_span<vertex_t const> const>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  std::vector<ValueIterator> edge_partition_value_firsts_{};
  std::vector<vertex_t> edge_partition_major_range_firsts_{};
};

template <typename vertex_t, typename ValueIterator>
class edge_minor_property_view_t {
 public:
  using value_type     = typename thrust::iterator_traits<ValueIterator>::value_type;
  using value_iterator = ValueIterator;

  edge_minor_property_view_t() = default;

  edge_minor_property_view_t(ValueIterator value_first, vertex_t minor_range_first)
    : value_first_(value_first), minor_range_first_(minor_range_first)
  {
  }

  edge_minor_property_view_t(raft::device_span<vertex_t const> keys,
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

  std::optional<raft::device_span<vertex_t const>> keys() const { return keys_; }

  std::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets() const
  {
    return key_chunk_start_offsets_;
  }

  std::optional<size_t> key_chunk_size() const { return key_chunk_size_; }

  ValueIterator value_first() const { return value_first_; }

  vertex_t minor_range_first() const { return minor_range_first_; }

 private:
  std::optional<raft::device_span<vertex_t const>> keys_{std::nullopt};
  std::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  ValueIterator value_first_{};
  vertex_t minor_range_first_{};
};

template <typename vertex_t, typename T>
class edge_major_property_t {
 public:
  using buffer_type = decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}));

  edge_major_property_t(raft::handle_t const& handle) {}

  edge_major_property_t(raft::handle_t const& handle,
                        std::vector<vertex_t> const& edge_partition_major_range_sizes,
                        std::vector<vertex_t> const& edge_partition_major_range_firsts)
    : edge_partition_major_range_firsts_(edge_partition_major_range_firsts)
  {
    buffers_.reserve(edge_partition_major_range_firsts_.size());
    for (size_t i = 0; i < edge_partition_major_range_firsts_.size(); ++i) {
      buffers_.push_back(
        allocate_dataframe_buffer<T>(edge_partition_major_range_sizes[i], handle.get_stream()));
    }
  }

  edge_major_property_t(
    raft::handle_t const& handle,
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_keys,
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_key_chunk_start_offsets,
    size_t key_chunk_size,
    std::vector<vertex_t> const& edge_partition_major_range_firsts)
    : edge_partition_keys_(edge_partition_keys),
      edge_partition_key_chunk_start_offsets_(edge_partition_key_chunk_start_offsets),
      key_chunk_size_(key_chunk_size),
      edge_partition_major_range_firsts_(edge_partition_major_range_firsts)
  {
    buffers_.reserve(edge_partition_major_range_firsts_.size());
    for (size_t i = 0; i < edge_partition_major_range_firsts_.size(); ++i) {
      buffers_.push_back(
        allocate_dataframe_buffer<T>(edge_partition_keys[i].size(), handle.get_stream()));
    }
  }

  void clear(raft::handle_t const& handle)
  {
    edge_partition_keys_                    = std::nullopt;
    edge_partition_key_chunk_start_offsets_ = std::nullopt;
    key_chunk_size_                         = std::nullopt;

    buffers_.clear();
    buffers_.shrink_to_fit();
    edge_partition_major_range_firsts_.clear();
    edge_partition_major_range_firsts_.shrink_to_fit();
  }

  auto view() const
  {
    using const_value_iterator = decltype(get_dataframe_buffer_cbegin(buffers_[0]));

    std::vector<const_value_iterator> edge_partition_value_firsts(buffers_.size());
    for (size_t i = 0; i < edge_partition_value_firsts.size(); ++i) {
      edge_partition_value_firsts[i] = get_dataframe_buffer_cbegin(buffers_[i]);
    }

    if (edge_partition_keys_) {
      return edge_major_property_view_t<vertex_t, const_value_iterator>(
        *edge_partition_keys_,
        *edge_partition_key_chunk_start_offsets_,
        *key_chunk_size_,
        edge_partition_value_firsts,
        edge_partition_major_range_firsts_);
    } else {
      return edge_major_property_view_t<vertex_t, const_value_iterator>(
        edge_partition_value_firsts, edge_partition_major_range_firsts_);
    }
  }

  auto mutable_view()
  {
    using value_iterator = decltype(get_dataframe_buffer_begin(buffers_[0]));

    std::vector<value_iterator> edge_partition_value_firsts(buffers_.size());
    for (size_t i = 0; i < edge_partition_value_firsts.size(); ++i) {
      edge_partition_value_firsts[i] = get_dataframe_buffer_begin(buffers_[i]);
    }

    if (edge_partition_keys_) {
      return edge_major_property_view_t<vertex_t, value_iterator>(
        *edge_partition_keys_,
        *edge_partition_key_chunk_start_offsets_,
        *key_chunk_size_,
        edge_partition_value_firsts,
        edge_partition_major_range_firsts_);
    } else {
      return edge_major_property_view_t<vertex_t, value_iterator>(
        edge_partition_value_firsts, edge_partition_major_range_firsts_);
    }
  }

 private:
  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> edge_partition_keys_{
    std::nullopt};
  std::optional<raft::host_span<raft::device_span<vertex_t const> const>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  std::vector<buffer_type> buffers_{};
  std::vector<vertex_t> edge_partition_major_range_firsts_{};
};

template <typename vertex_t, typename T>
class edge_minor_property_t {
 public:
  edge_minor_property_t(raft::handle_t const& handle)
    : buffer_(allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream())),
      minor_range_first_(vertex_t{0})
  {
  }

  edge_minor_property_t(raft::handle_t const& handle,
                        vertex_t buffer_size,
                        vertex_t minor_range_first)
    : buffer_(allocate_dataframe_buffer<T>(buffer_size, handle.get_stream())),
      minor_range_first_(minor_range_first)
  {
  }

  edge_minor_property_t(raft::handle_t const& handle,
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

  auto view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);
    if (keys_) {
      return edge_minor_property_view_t<vertex_t, decltype(value_first)>(
        *keys_, *key_chunk_start_offsets_, *key_chunk_size_, value_first, minor_range_first_);
    } else {
      return edge_minor_property_view_t<vertex_t, decltype(value_first)>(value_first,
                                                                         minor_range_first_);
    }
  }

  auto mutable_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);
    if (keys_) {
      return edge_minor_property_view_t<vertex_t, decltype(value_first)>(
        *keys_, *key_chunk_start_offsets_, *key_chunk_size_, value_first, minor_range_first_);
    } else {
      return edge_minor_property_view_t<vertex_t, decltype(value_first)>(value_first,
                                                                         minor_range_first_);
    }
  }

 private:
  std::optional<raft::device_span<vertex_t const>> keys_{std::nullopt};
  std::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{})) buffer_;
  vertex_t minor_range_first_{};
};

class edge_endpoint_dummy_property_view_t {
 public:
  using value_type     = thrust::nullopt_t;
  using value_iterator = void*;
};

}  // namespace detail

template <typename GraphViewType, typename T>
class edge_src_property_t {
 public:
  using value_type = T;
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  edge_src_property_t(raft::handle_t const& handle) : property_(handle) {}

  edge_src_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
    : property_(handle)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    auto key_chunk_size = graph_view.local_sorted_unique_edge_src_chunk_size();
    if (key_chunk_size) {
      if constexpr (GraphViewType::is_multi_gpu) {
        if constexpr (GraphViewType::is_storage_transposed) {
          property_ = detail::edge_minor_property_t<vertex_t, T>(
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
          std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
          for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
            (*edge_partition_keys_)[i] = *(graph_view.local_sorted_unique_edge_srcs(i));
            (*edge_partition_key_chunk_start_offsets_)[i] =
              *(graph_view.local_sorted_unique_edge_src_chunk_start_offsets(i));
            major_range_firsts[i] = graph_view.local_edge_partition_src_range_first(i);
          }
          property_ = detail::edge_major_property_t<vertex_t, T>(
            handle,
            raft::host_span<raft::device_span<vertex_t const>>((*edge_partition_keys_).data(),
                                                               (*edge_partition_keys_).size()),
            raft::host_span<raft::device_span<vertex_t const>>(
              (*edge_partition_key_chunk_start_offsets_).data(),
              (*edge_partition_key_chunk_start_offsets_).size()),
            *key_chunk_size,
            std::move(major_range_firsts));
        }
      } else {
        assert(false);
      }
    } else {
      if constexpr (GraphViewType::is_storage_transposed) {
        property_ = detail::edge_minor_property_t<vertex_t, T>(
          handle,
          graph_view.local_edge_partition_src_range_size(),
          graph_view.local_edge_partition_src_range_first());
      } else {
        std::vector<vertex_t> major_range_sizes(graph_view.number_of_local_edge_partitions(),
                                                vertex_t{0});
        std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          major_range_sizes[i]  = graph_view.local_edge_partition_src_range_size(i);
          major_range_firsts[i] = graph_view.local_edge_partition_src_range_first(i);
        }
        property_ = detail::edge_major_property_t<vertex_t, T>(
          handle, std::move(major_range_sizes), std::move(major_range_firsts));
      }
    }
  }

  void clear(raft::handle_t const& handle)
  {
    property_.clear(handle);

    edge_partition_keys_                    = std::nullopt;
    edge_partition_key_chunk_start_offsets_ = std::nullopt;
  }

  auto view() const { return property_.view(); }
  auto mutable_view() { return property_.mutable_view(); }

 private:
  std::conditional_t<GraphViewType::is_storage_transposed,
                     detail::edge_minor_property_t<typename GraphViewType::vertex_type, T>,
                     detail::edge_major_property_t<typename GraphViewType::vertex_type, T>>
    property_;

  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_keys_{std::nullopt};
  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
};

template <typename GraphViewType, typename T>
class edge_dst_property_t {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  edge_dst_property_t(raft::handle_t const& handle) : property_(handle) {}

  edge_dst_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
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
          std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
          for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
            (*edge_partition_keys_)[i] = *(graph_view.local_sorted_unique_edge_dsts(i));
            (*edge_partition_key_chunk_start_offsets_)[i] =
              *(graph_view.local_sorted_unique_edge_dst_chunk_start_offsets(i));
            major_range_firsts[i] = graph_view.local_edge_partition_dst_range_first(i);
          }
          property_ = detail::edge_major_property_t<vertex_t, T>(
            handle,
            raft::host_span<raft::device_span<vertex_t const>>((*edge_partition_keys_).data(),
                                                               (*edge_partition_keys_).size()),
            raft::host_span<raft::device_span<vertex_t const>>(
              (*edge_partition_key_chunk_start_offsets_).data(),
              (*edge_partition_key_chunk_start_offsets_).size()),
            *key_chunk_size,
            std::move(major_range_firsts));
        } else {
          property_ = detail::edge_minor_property_t<vertex_t, T>(
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
        std::vector<vertex_t> major_range_sizes(graph_view.number_of_local_edge_partitions(),
                                                vertex_t{0});
        std::vector<vertex_t> major_range_firsts(graph_view.number_of_local_edge_partitions());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          major_range_sizes[i]  = graph_view.local_edge_partition_dst_range_size(i);
          major_range_firsts[i] = graph_view.local_edge_partition_dst_range_first(i);
        }
        property_ = detail::edge_major_property_t<vertex_t, T>(
          handle, std::move(major_range_sizes), std::move(major_range_firsts));
      } else {
        property_ = detail::edge_minor_property_t<vertex_t, T>(
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

  auto view() const { return property_.view(); }
  auto mutable_view() { return property_.mutable_view(); }

 private:
  std::conditional_t<GraphViewType::is_storage_transposed,
                     detail::edge_major_property_t<typename GraphViewType::vertex_type, T>,
                     detail::edge_minor_property_t<typename GraphViewType::vertex_type, T>>
    property_;

  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_keys_{std::nullopt};
  std::optional<std::vector<raft::device_span<typename GraphViewType::vertex_type const>>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
};

class edge_src_dummy_property_t {
 public:
  using value_type = thrust::nullopt_t;

  auto view() const { return detail::edge_endpoint_dummy_property_view_t{}; }
};

class edge_dst_dummy_property_t {
 public:
  using value_type = thrust::nullopt_t;

  auto view() const { return detail::edge_endpoint_dummy_property_view_t{}; }
};

template <typename vertex_t, typename... Ts>
auto view_concat(detail::edge_major_property_view_t<vertex_t, Ts> const&... views)
{
  using concat_value_iterator = decltype(thrust::make_zip_iterator(
    thrust_tuple_cat(detail::to_thrust_tuple(views.value_firsts()[0])...)));

  std::vector<concat_value_iterator> edge_partition_concat_value_firsts{};
  auto first_view = detail::get_first_of_pack(views...);
  edge_partition_concat_value_firsts.resize(first_view.major_range_firsts().size());
  for (size_t i = 0; i < edge_partition_concat_value_firsts.size(); ++i) {
    edge_partition_concat_value_firsts[i] = thrust::make_zip_iterator(
      thrust_tuple_cat(detail::to_thrust_tuple(views.value_firsts()[i])...));
  }

  if (first_view.key_chunk_size()) {
    return detail::edge_major_property_view_t<vertex_t, concat_value_iterator>(
      *(first_view.keys()),
      *(first_view.key_chunk_start_offsets()),
      *(first_view.key_chunk_size()),
      edge_partition_concat_value_firsts,
      first_view.major_range_firsts());
  } else {
    return detail::edge_major_property_view_t<vertex_t, concat_value_iterator>(
      edge_partition_concat_value_firsts, first_view.major_range_firsts());
  }
}

template <typename vertex_t, typename... Ts>
auto view_concat(detail::edge_minor_property_view_t<vertex_t, Ts> const&... views)
{
  using concat_value_iterator = decltype(
    thrust::make_zip_iterator(thrust_tuple_cat(detail::to_thrust_tuple(views.value_first())...)));

  concat_value_iterator edge_partition_concat_value_first{};

  auto first_view = detail::get_first_of_pack(views...);

  edge_partition_concat_value_first =
    thrust::make_zip_iterator(thrust_tuple_cat(detail::to_thrust_tuple(views.value_first())...));

  if (first_view.key_chunk_size()) {
    return detail::edge_minor_property_view_t<vertex_t, concat_value_iterator>(
      *(first_view.keys()),
      *(first_view.key_chunk_start_offsets()),
      *(first_view.key_chunk_size()),
      edge_partition_concat_value_first,
      first_view.minor_range_first());
  } else {
    return detail::edge_minor_property_view_t<vertex_t, concat_value_iterator>(
      edge_partition_concat_value_first, first_view.minor_range_first());
  }
}

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>

#include <optional>
#include <type_traits>
#include <variant>

namespace cugraph {

namespace detail {

template <typename vertex_t>
struct edge_major_key_info_t {
  raft::host_span<raft::device_span<vertex_t const> const> edge_partition_keys{};
  raft::host_span<raft::device_span<vertex_t const> const> edge_partition_key_chunk_start_offsets{};
  size_t key_chunk_size{};
};

template <typename vertex_t>
struct edge_minor_key_info_t {
  raft::device_span<vertex_t const> keys{};
  raft::device_span<vertex_t const> key_chunk_start_offsets{};
  size_t key_chunk_size{};
};

template <typename vertex_t,
          typename ValueIterator,
          typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
class edge_endpoint_property_view_t {
 public:
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t> ||
    cugraph::has_packed_bool_element<ValueIterator, value_t>());

  using vertex_type    = vertex_t;
  using value_type     = value_t;
  using value_iterator = ValueIterator;

  edge_endpoint_property_view_t() = default;

  // major
  edge_endpoint_property_view_t(std::vector<ValueIterator> const& edge_partition_value_firsts,
                                std::vector<vertex_t> const& edge_partition_range_firsts)
    : value_firsts_(edge_partition_value_firsts), range_firsts_(edge_partition_range_firsts)
  {
  }

  // major
  edge_endpoint_property_view_t(
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_keys,
    raft::host_span<raft::device_span<vertex_t const> const> edge_partition_key_chunk_start_offsets,
    size_t key_chunk_size,
    std::vector<ValueIterator> const& edge_partition_value_firsts,
    std::vector<vertex_t> const& edge_partition_range_firsts)
    : key_info_(edge_major_key_info_t<vertex_t>{
        edge_partition_keys, edge_partition_key_chunk_start_offsets, key_chunk_size}),
      value_firsts_(edge_partition_value_firsts),
      range_firsts_(edge_partition_range_firsts)
  {
  }

  // minor
  edge_endpoint_property_view_t(ValueIterator value_first, vertex_t range_first)
    : value_firsts_(value_first), range_firsts_(range_first)
  {
  }

  // minor
  edge_endpoint_property_view_t(raft::device_span<vertex_t const> keys,
                                raft::device_span<vertex_t const> key_chunk_start_offsets,
                                size_t key_chunk_size,
                                ValueIterator value_first,
                                vertex_t range_first)
    : key_info_(edge_minor_key_info_t<vertex_t>{keys, key_chunk_start_offsets, key_chunk_size}),
      value_firsts_(value_first),
      range_firsts_(range_first)
  {
  }

  ~edge_endpoint_property_view_t()
  {  // to silence a spurious "maybe used uninitialized" warning
    key_info_ = std::nullopt;
    if (value_firsts_.index() == 0) {
      std::get<0>(value_firsts_).clear();
      std::get<0>(value_firsts_).shrink_to_fit();
    }
    if (range_firsts_.index() == 0) {
      std::get<0>(range_firsts_).clear();
      std::get<0>(range_firsts_).shrink_to_fit();
    }
  }

  std::optional<raft::host_span<raft::device_span<vertex_t const> const>> major_keys() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 0,
                    "This function is valid only when this object stores a view object of edge "
                    "major property values.");
    return key_info_ ? std::make_optional(std::get<0>(*key_info_).edge_partition_keys)
                     : std::nullopt;
  }

  std::optional<raft::host_span<raft::device_span<vertex_t const> const>>
  major_key_chunk_start_offsets() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 0,
                    "This function is valid only when this object stores a view object of edge "
                    "major property values.");
    return key_info_
             ? std::make_optional(std::get<0>(*key_info_).edge_partition_key_chunk_start_offsets)
             : std::nullopt;
  }

  std::optional<raft::device_span<vertex_t const>> minor_keys() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 1,
                    "This function is valid only when this object stores a view object of edge "
                    "minor property values.");
    return key_info_ ? std::make_optional(std::get<1>(*key_info_).keys) : std::nullopt;
  }

  std::optional<raft::device_span<vertex_t const>> minor_key_chunk_start_offsets() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 1,
                    "This function is valid only when this object stores a view object of edge "
                    "minor property values.");
    return key_info_ ? std::make_optional(std::get<1>(*key_info_).key_chunk_start_offsets)
                     : std::nullopt;
  }

  std::optional<size_t> key_chunk_size() const
  {
    if (key_info_) {
      return key_info_->index() == 0 ? std::get<0>(*key_info_).key_chunk_size
                                     : std::get<1>(*key_info_).key_chunk_size;
    } else {
      return std::nullopt;
    }
  }

  std::vector<ValueIterator> const& major_value_firsts() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 0,
                    "This function is valid only when this object stores a view object of edge "
                    "major property values.");
    return std::get<0>(value_firsts_);
  }

  std::vector<vertex_t> const& major_range_firsts() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 0,
                    "This function is valid only when this object stores a view object of edge "
                    "major property values.");
    return std::get<0>(range_firsts_);
  }

  ValueIterator minor_value_first() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 1,
                    "This function is valid only when this object stores a view object of edge "
                    "minor property values.");
    return std::get<1>(value_firsts_);
  }

  vertex_t minor_range_first() const
  {
    CUGRAPH_EXPECTS(value_firsts_.index() == 1,
                    "This function is valid only when this object stores a view object of edge "
                    "minor property values.");
    return std::get<1>(range_firsts_);
  }

  bool is_major() const { return (value_firsts_.index() == 0) ? true : false; }

 private:
  std::optional<std::variant<edge_major_key_info_t<vertex_t>, edge_minor_key_info_t<vertex_t>>>
    key_info_{std::nullopt};

  std::variant<std::vector<ValueIterator> /* major */, ValueIterator /* minor */> value_firsts_{};
  std::variant<std::vector<vertex_t> /* major */, vertex_t /* minor */> range_firsts_{};
};

template <typename vertex_t, typename T>
class edge_major_property_t {
 public:
  static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using buffer_type =
    dataframe_buffer_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;
  using value_iterator =
    dataframe_buffer_iterator_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;
  using const_value_iterator = dataframe_buffer_const_iterator_type_t<
    std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;

  template <typename GraphViewType>
  edge_major_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    static_assert(std::is_same_v<vertex_t, typename GraphViewType::vertex_type>);

    key_chunk_size_ = GraphViewType::is_storage_transposed
                        ? graph_view.local_sorted_unique_edge_dst_chunk_size()
                        : graph_view.local_sorted_unique_edge_src_chunk_size();
    buffers_.reserve(graph_view.number_of_local_edge_partitions());
    edge_partition_major_range_firsts_.resize(graph_view.number_of_local_edge_partitions());
    if (key_chunk_size_) {
      assert(GraphViewType::is_multi_gpu);
      edge_partition_keys_ = std::vector<raft::device_span<vertex_t const>>(
        graph_view.number_of_local_edge_partitions());
      edge_partition_key_chunk_start_offsets_ = std::vector<raft::device_span<vertex_t const>>(
        graph_view.number_of_local_edge_partitions());
      for (size_t i = 0; i < edge_partition_major_range_firsts_.size(); ++i) {
        if constexpr (GraphViewType::is_storage_transposed) {
          edge_partition_keys_->operator[](i) = *(graph_view.local_sorted_unique_edge_dsts(i));
          edge_partition_key_chunk_start_offsets_->operator[](i) =
            *(graph_view.local_sorted_unique_edge_dst_chunk_start_offsets(i));
          edge_partition_major_range_firsts_[i] =
            graph_view.local_edge_partition_dst_range_first(i);
          edge_partition_major_range_firsts_[i] =
            graph_view.local_edge_partition_dst_range_first(i);
        } else {
          edge_partition_keys_->operator[](i) = *(graph_view.local_sorted_unique_edge_srcs(i));
          edge_partition_key_chunk_start_offsets_->operator[](i) =
            *(graph_view.local_sorted_unique_edge_src_chunk_start_offsets(i));
          edge_partition_major_range_firsts_[i] =
            graph_view.local_edge_partition_src_range_first(i);
          edge_partition_major_range_firsts_[i] =
            graph_view.local_edge_partition_src_range_first(i);
        }
        buffers_.push_back(
          allocate_dataframe_buffer<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>(
            std::is_same_v<T, bool>
              ? cugraph::packed_bool_size(edge_partition_keys_->operator[](i).size())
              : edge_partition_keys_->operator[](i).size(),
            handle.get_stream()));
      }
    } else {
      for (size_t i = 0; i < edge_partition_major_range_firsts_.size(); ++i) {
        vertex_t range_size{};
        if constexpr (GraphViewType::is_storage_transposed) {
          range_size = graph_view.local_edge_partition_dst_range_size(i);
          edge_partition_major_range_firsts_[i] =
            graph_view.local_edge_partition_dst_range_first(i);
        } else {
          range_size = graph_view.local_edge_partition_src_range_size(i);
          edge_partition_major_range_firsts_[i] =
            graph_view.local_edge_partition_src_range_first(i);
        }
        buffers_.push_back(
          allocate_dataframe_buffer<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>(
            std::is_same_v<T, bool> ? cugraph::packed_bool_size(static_cast<size_t>(range_size))
                                    : static_cast<size_t>(range_size),
            handle.get_stream()));
      }
    }
  }

  edge_major_property_t()                                        = delete;
  edge_major_property_t(edge_major_property_t&&)                 = default;
  edge_major_property_t(edge_major_property_t const&)            = delete;
  edge_major_property_t& operator=(edge_major_property_t&&)      = default;
  edge_major_property_t& operator=(edge_major_property_t const&) = delete;
  ~edge_major_property_t()
  {  // to silence a spurious "maybe used uninitialized" warning
    this->clear();
  }

  void clear()
  {
    edge_partition_keys_.reset();
    edge_partition_key_chunk_start_offsets_.reset();
    key_chunk_size_.reset();

    buffers_.clear();
    buffers_.shrink_to_fit();
    edge_partition_major_range_firsts_.clear();
    edge_partition_major_range_firsts_.shrink_to_fit();
  }

  auto view() const
  {
    std::vector<const_value_iterator> edge_partition_value_firsts(buffers_.size());
    for (size_t i = 0; i < edge_partition_value_firsts.size(); ++i) {
      edge_partition_value_firsts[i] = get_dataframe_buffer_cbegin(buffers_[i]);
    }

    if (edge_partition_keys_) {
      return edge_endpoint_property_view_t<vertex_t, const_value_iterator, T>(
        raft::host_span<raft::device_span<vertex_t const> const>(edge_partition_keys_->data(),
                                                                 edge_partition_keys_->size()),
        raft::host_span<raft::device_span<vertex_t const> const>(
          edge_partition_key_chunk_start_offsets_->data(),
          edge_partition_key_chunk_start_offsets_->size()),
        *key_chunk_size_,
        edge_partition_value_firsts,
        edge_partition_major_range_firsts_);
    } else {
      return edge_endpoint_property_view_t<vertex_t, const_value_iterator, T>(
        edge_partition_value_firsts, edge_partition_major_range_firsts_);
    }
  }

  auto mutable_view()
  {
    std::vector<value_iterator> edge_partition_value_firsts(buffers_.size());
    for (size_t i = 0; i < edge_partition_value_firsts.size(); ++i) {
      edge_partition_value_firsts[i] = get_dataframe_buffer_begin(buffers_[i]);
    }

    if (edge_partition_keys_) {
      return edge_endpoint_property_view_t<vertex_t, value_iterator, T>(
        raft::host_span<raft::device_span<vertex_t const>>(edge_partition_keys_->data(),
                                                           edge_partition_keys_->size()),
        raft::host_span<raft::device_span<vertex_t const>>(
          edge_partition_key_chunk_start_offsets_->data(),
          edge_partition_key_chunk_start_offsets_->size()),
        *key_chunk_size_,
        edge_partition_value_firsts,
        edge_partition_major_range_firsts_);
    } else {
      return edge_endpoint_property_view_t<vertex_t, value_iterator, T>(
        edge_partition_value_firsts, edge_partition_major_range_firsts_);
    }
  }

 private:
  std::optional<std::vector<raft::device_span<vertex_t const>>> edge_partition_keys_{std::nullopt};
  std::optional<std::vector<raft::device_span<vertex_t const>>>
    edge_partition_key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  std::vector<buffer_type> buffers_{};
  std::vector<vertex_t> edge_partition_major_range_firsts_{};
};

template <typename vertex_t, typename T>
class edge_minor_property_t {
 public:
  static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using value_iterator =
    dataframe_buffer_iterator_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;
  using const_value_iterator = dataframe_buffer_const_iterator_type_t<
    std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;

  template <typename GraphViewType>
  edge_minor_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
    : buffer_(allocate_dataframe_buffer<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>(
        0, handle.get_stream()))
  {
    static_assert(std::is_same_v<vertex_t, typename GraphViewType::vertex_type>);

    if constexpr (GraphViewType::is_storage_transposed) {
      key_chunk_size_    = graph_view.local_sorted_unique_edge_src_chunk_size();
      minor_range_first_ = graph_view.local_edge_partition_src_range_first();
    } else {
      key_chunk_size_    = graph_view.local_sorted_unique_edge_dst_chunk_size();
      minor_range_first_ = graph_view.local_edge_partition_dst_range_first();
    }
    if (key_chunk_size_) {
      assert(GraphViewType::is_multi_gpu);
      if constexpr (GraphViewType::is_storage_transposed) {
        keys_                    = *(graph_view.local_sorted_unique_edge_srcs());
        key_chunk_start_offsets_ = *(graph_view.local_sorted_unique_edge_src_chunk_start_offsets());
      } else {
        keys_                    = *(graph_view.local_sorted_unique_edge_dsts());
        key_chunk_start_offsets_ = *(graph_view.local_sorted_unique_edge_dst_chunk_start_offsets());
      }
      resize_dataframe_buffer(
        buffer_,
        std::is_same_v<T, bool> ? cugraph::packed_bool_size(keys_->size()) : keys_->size(),
        handle.get_stream());
    } else {
      vertex_t range_size{};
      if constexpr (GraphViewType::is_storage_transposed) {
        range_size = graph_view.local_edge_partition_src_range_size();
      } else {
        range_size = graph_view.local_edge_partition_dst_range_size();
      }
      resize_dataframe_buffer(buffer_,
                              std::is_same_v<T, bool>
                                ? cugraph::packed_bool_size(static_cast<size_t>(range_size))
                                : static_cast<size_t>(range_size),
                              handle.get_stream());
    }
  }

  edge_minor_property_t()                                        = delete;
  edge_minor_property_t(edge_minor_property_t&&)                 = default;
  edge_minor_property_t(edge_minor_property_t const&)            = delete;
  edge_minor_property_t& operator=(edge_minor_property_t&&)      = default;
  edge_minor_property_t& operator=(edge_minor_property_t const&) = delete;
  ~edge_minor_property_t()
  {  // to silence a spurious "maybe used uninitialized" warning
    this->clear();
  }

  void clear()
  {
    keys_.reset();
    key_chunk_start_offsets_.reset();
    key_chunk_size_.reset();

    rmm::cuda_stream_view stream{};
    if constexpr (std::is_arithmetic_v<T>) {
      stream = buffer_.stream();
    } else {
      stream = std::get<0>(buffer_).stream();
    }
    resize_dataframe_buffer(buffer_, size_t{0}, stream);
    shrink_to_fit_dataframe_buffer(buffer_, stream);
    minor_range_first_ = vertex_t{0};
  }

  auto view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);
    if (keys_) {
      return edge_endpoint_property_view_t<vertex_t, decltype(value_first), T>(
        *keys_, *key_chunk_start_offsets_, *key_chunk_size_, value_first, minor_range_first_);
    } else {
      return edge_endpoint_property_view_t<vertex_t, decltype(value_first), T>(value_first,
                                                                               minor_range_first_);
    }
  }

  auto mutable_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);
    if (keys_) {
      return edge_endpoint_property_view_t<vertex_t, decltype(value_first), T>(
        *keys_, *key_chunk_start_offsets_, *key_chunk_size_, value_first, minor_range_first_);
    } else {
      return edge_endpoint_property_view_t<vertex_t, decltype(value_first), T>(value_first,
                                                                               minor_range_first_);
    }
  }

 private:
  std::optional<raft::device_span<vertex_t const>> keys_{std::nullopt};
  std::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{std::nullopt};
  std::optional<size_t> key_chunk_size_{std::nullopt};

  dataframe_buffer_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>> buffer_;
  vertex_t minor_range_first_{};
};

class edge_endpoint_dummy_property_view_t {
 public:
  using value_type     = cuda::std::nullopt_t;
  using value_iterator = void*;
};

template <typename vertex_t, typename... Iters, typename... Types>
auto major_view_concat(
  detail::edge_endpoint_property_view_t<vertex_t, Iters, Types> const&... views)
{
  using concat_value_iterator = decltype(thrust::make_zip_iterator(
    thrust_tuple_cat(to_thrust_iterator_tuple(views.major_value_firsts()[0])...)));
  using concat_value_type     = decltype(thrust_tuple_cat(to_thrust_tuple(Types{})...));

  std::vector<concat_value_iterator> edge_partition_concat_value_firsts{};
  auto first_view = get_first_of_pack(views...);
  edge_partition_concat_value_firsts.resize(first_view.major_range_firsts().size());
  for (size_t i = 0; i < edge_partition_concat_value_firsts.size(); ++i) {
    edge_partition_concat_value_firsts[i] = thrust::make_zip_iterator(
      thrust_tuple_cat(to_thrust_iterator_tuple(views.major_value_firsts()[i])...));
  }

  if (first_view.key_chunk_size()) {
    return detail::
      edge_endpoint_property_view_t<vertex_t, concat_value_iterator, concat_value_type>(
        *(first_view.major_keys()),
        *(first_view.major_key_chunk_start_offsets()),
        *(first_view.key_chunk_size()),
        edge_partition_concat_value_firsts,
        first_view.major_range_firsts());
  } else {
    return detail::
      edge_endpoint_property_view_t<vertex_t, concat_value_iterator, concat_value_type>(
        edge_partition_concat_value_firsts, first_view.major_range_firsts());
  }
}

template <typename vertex_t, typename... Iters, typename... Types>
auto minor_view_concat(
  detail::edge_endpoint_property_view_t<vertex_t, Iters, Types> const&... views)
{
  using concat_value_iterator = decltype(thrust::make_zip_iterator(
    thrust_tuple_cat(to_thrust_iterator_tuple(views.minor_value_first())...)));
  using concat_value_type     = decltype(thrust_tuple_cat(to_thrust_tuple(Types{})...));

  concat_value_iterator edge_partition_concat_value_first{};

  auto first_view = get_first_of_pack(views...);

  edge_partition_concat_value_first = thrust::make_zip_iterator(
    thrust_tuple_cat(to_thrust_iterator_tuple(views.minor_value_first())...));

  if (first_view.key_chunk_size()) {
    return detail::
      edge_endpoint_property_view_t<vertex_t, concat_value_iterator, concat_value_type>(
        *(first_view.minor_keys()),
        *(first_view.minor_key_chunk_start_offsets()),
        *(first_view.key_chunk_size()),
        edge_partition_concat_value_first,
        first_view.minor_range_first());
  } else {
    return detail::
      edge_endpoint_property_view_t<vertex_t, concat_value_iterator, concat_value_type>(
        edge_partition_concat_value_first, first_view.minor_range_first());
  }
}

}  // namespace detail

template <typename vertex_t, typename T>
class edge_src_property_t {
 public:
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using value_type = T;
  using value_iterator =
    dataframe_buffer_iterator_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;
  using const_value_iterator = dataframe_buffer_const_iterator_type_t<
    std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;

  edge_src_property_t(raft::handle_t const& handle) {}

  template <typename GraphViewType>
  edge_src_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    static_assert(std::is_same_v<vertex_t, typename GraphViewType::vertex_type>);
    if constexpr (GraphViewType::is_storage_transposed) {
      property_ = detail::edge_minor_property_t<vertex_t, T>(handle, graph_view);
    } else {
      property_ = detail::edge_major_property_t<vertex_t, T>(handle, graph_view);
    }
  }

  edge_src_property_t()                                      = delete;
  edge_src_property_t(edge_src_property_t&&)                 = default;
  edge_src_property_t(edge_src_property_t const&)            = delete;
  edge_src_property_t& operator=(edge_src_property_t&&)      = default;
  edge_src_property_t& operator=(edge_src_property_t const&) = delete;
  ~edge_src_property_t()
  {
    this->clear();
  }  // to silence a spurious "maybe used uninitialized" warning

  void clear() { property_ = std::monostate{}; }

  auto view() const
  {
    CUGRAPH_EXPECTS(property_.index() != 0,
                    "This function should not be called before initialization.");
    return property_.index() == 1 ? std::get<1>(property_).view() : std::get<2>(property_).view();
  }

  auto mutable_view()
  {
    CUGRAPH_EXPECTS(property_.index() != 0,
                    "This function should not be called before initialization.");
    return property_.index() == 1 ? std::get<1>(property_).mutable_view()
                                  : std::get<2>(property_).mutable_view();
  }

 private:
  std::variant<std::monostate,
               detail::edge_major_property_t<vertex_t, T>,
               detail::edge_minor_property_t<vertex_t, T>>
    property_{};
};

template <typename vertex_t, typename T>
class edge_dst_property_t {
 public:
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using value_type = T;
  using value_iterator =
    dataframe_buffer_iterator_type_t<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;
  using const_value_iterator = dataframe_buffer_const_iterator_type_t<
    std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>;

  edge_dst_property_t(raft::handle_t const& handle) {}

  template <typename GraphViewType>
  edge_dst_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    static_assert(std::is_same_v<vertex_t, typename GraphViewType::vertex_type>);
    if constexpr (GraphViewType::is_storage_transposed) {
      property_ = detail::edge_major_property_t<vertex_t, T>(handle, graph_view);
    } else {
      property_ = detail::edge_minor_property_t<vertex_t, T>(handle, graph_view);
    }
  }

  edge_dst_property_t()                                      = delete;
  edge_dst_property_t(edge_dst_property_t&&)                 = default;
  edge_dst_property_t(edge_dst_property_t const&)            = delete;
  edge_dst_property_t& operator=(edge_dst_property_t&&)      = default;
  edge_dst_property_t& operator=(edge_dst_property_t const&) = delete;
  ~edge_dst_property_t()
  {
    this->clear();
  }  // to silence a spurious "maybe used uninitialized" warning

  void clear() { property_ = std::monostate{}; }

  auto view() const
  {
    CUGRAPH_EXPECTS(property_.index() != 0,
                    "This function should not be called before initialization.");
    return property_.index() == 1 ? std::get<1>(property_).view() : std::get<2>(property_).view();
  }

  auto mutable_view()
  {
    CUGRAPH_EXPECTS(property_.index() != 0,
                    "This function should not be called before initialization.");
    return property_.index() == 1 ? std::get<1>(property_).mutable_view()
                                  : std::get<2>(property_).mutable_view();
  }

 private:
  std::variant<std::monostate,
               detail::edge_major_property_t<vertex_t, T>,
               detail::edge_minor_property_t<vertex_t, T>>
    property_{};
};

class edge_src_dummy_property_t {
 public:
  using value_type = cuda::std::nullopt_t;

  auto view() const { return detail::edge_endpoint_dummy_property_view_t{}; }
};

class edge_dst_dummy_property_t {
 public:
  using value_type = cuda::std::nullopt_t;

  auto view() const { return detail::edge_endpoint_dummy_property_view_t{}; }
};

// SG-only (use a vertex property buffer instead of creating a new edge_src_property_t object to
// save memory)
template <typename vertex_t, typename T, typename GraphViewType>
detail::edge_endpoint_property_view_t<vertex_t,
                                      typename edge_src_property_t<vertex_t,
                                                                   T>::const_value_iterator,
                                      T>
make_edge_src_property_view(
  GraphViewType const& graph_view,
  typename edge_src_property_t<vertex_t, T>::const_value_iterator value_first,
  size_t num_values)
{
  using const_value_iterator = typename edge_src_property_t<vertex_t, T>::const_value_iterator;

  CUGRAPH_EXPECTS(!GraphViewType::is_multi_gpu,
                  "Invalid input argument: this function is only for single-GPU.");

  vertex_t range_size{};
  if constexpr (GraphViewType::is_storage_transposed) {  // minor
    range_size = graph_view.local_edge_partition_src_range_size();
  } else {  // major
    range_size = graph_view.local_edge_partition_src_range_size(size_t{0});
  }
  auto expected_num_values = std::is_same_v<T, bool>
                               ? cugraph::packed_bool_size(static_cast<size_t>(range_size))
                               : static_cast<size_t>(range_size);
  CUGRAPH_EXPECTS(
    num_values == expected_num_values,
    "Invalid input argument: num_values does not match the expected number of values.");

  if constexpr (GraphViewType::is_storage_transposed) {  // minor
    return detail::edge_endpoint_property_view_t<vertex_t, const_value_iterator, T>(value_first,
                                                                                    vertex_t{0});
  } else {  // major
    return detail::edge_endpoint_property_view_t<vertex_t, const_value_iterator, T>(
      std::vector<const_value_iterator>{value_first}, std::vector<vertex_t>{vertex_t{0}});
  }
}

// SG-only (use a vertex property buffer instead of creating a new edge_src_property_t object to
// save memory)
template <typename vertex_t, typename T, typename GraphViewType>
detail::edge_endpoint_property_view_t<vertex_t,
                                      typename edge_src_property_t<vertex_t, T>::value_iterator,
                                      T>
make_edge_src_property_mutable_view(
  GraphViewType const& graph_view,
  typename edge_src_property_t<vertex_t, T>::value_iterator value_first,
  size_t num_values)
{
  using value_iterator = typename edge_src_property_t<vertex_t, T>::value_iterator;

  CUGRAPH_EXPECTS(!GraphViewType::is_multi_gpu,
                  "Invalid input argument: this function is only for single-GPU.");

  vertex_t range_size{};
  if constexpr (GraphViewType::is_storage_transposed) {  // minor
    range_size = graph_view.local_edge_partition_src_range_size();
  } else {  // major
    range_size = graph_view.local_edge_partition_src_range_size(size_t{0});
  }
  auto expected_num_values = std::is_same_v<T, bool>
                               ? cugraph::packed_bool_size(static_cast<size_t>(range_size))
                               : static_cast<size_t>(range_size);
  CUGRAPH_EXPECTS(
    num_values == expected_num_values,
    "Invalid input argument: num_values does not match the expected number of values.");

  if constexpr (GraphViewType::is_storage_transposed) {  // minor
    return detail::edge_endpoint_property_view_t<vertex_t, value_iterator, T>(value_first,
                                                                              vertex_t{0});
  } else {  // major
    return detail::edge_endpoint_property_view_t<vertex_t, value_iterator, T>(
      std::vector<value_iterator>{value_first}, std::vector<vertex_t>{vertex_t{0}});
  }
}

// SG-only (use a vertex property buffer instead of creating a new edge_dst_property_t object to
// save memory)
template <typename vertex_t, typename T, typename GraphViewType>
detail::edge_endpoint_property_view_t<vertex_t,
                                      typename edge_dst_property_t<vertex_t,
                                                                   T>::const_value_iterator,
                                      T>
make_edge_dst_property_view(
  GraphViewType const& graph_view,
  typename edge_dst_property_t<vertex_t, T>::const_value_iterator value_first,
  size_t num_values)
{
  using const_value_iterator = typename edge_dst_property_t<vertex_t, T>::const_value_iterator;

  CUGRAPH_EXPECTS(!GraphViewType::is_multi_gpu,
                  "Invalid input argument: this function is only for single-GPU.");

  vertex_t range_size{};
  if constexpr (GraphViewType::is_storage_transposed) {  // major
    range_size = graph_view.local_edge_partition_dst_range_size(size_t{0});
  } else {  // minor
    range_size = graph_view.local_edge_partition_dst_range_size();
  }
  auto expected_num_values = std::is_same_v<T, bool>
                               ? cugraph::packed_bool_size(static_cast<size_t>(range_size))
                               : static_cast<size_t>(range_size);
  CUGRAPH_EXPECTS(
    num_values == expected_num_values,
    "Invalid input argument: num_values does not match the expected number of values.");

  if constexpr (GraphViewType::is_storage_transposed) {  // major
    return detail::edge_endpoint_property_view_t<vertex_t, const_value_iterator, T>(
      std::vector<const_value_iterator>{value_first}, std::vector<vertex_t>{vertex_t{0}});
  } else {  // minor
    return detail::edge_endpoint_property_view_t<vertex_t, const_value_iterator, T>(value_first,
                                                                                    vertex_t{0});
  }
}

// SG-only (use a vertex property buffer instead of creating a new edge_dst_property_t object to
// save memory)
template <typename vertex_t, typename T, typename GraphViewType>
detail::edge_endpoint_property_view_t<vertex_t,
                                      typename edge_dst_property_t<vertex_t, T>::value_iterator,
                                      T>
make_edge_dst_property_mutable_view(
  GraphViewType const& graph_view,
  typename edge_dst_property_t<vertex_t, T>::value_iterator value_first,
  size_t num_values)
{
  using value_iterator = typename edge_dst_property_t<vertex_t, T>::value_iterator;

  CUGRAPH_EXPECTS(!GraphViewType::is_multi_gpu,
                  "Invalid input argument: this function is only for single-GPU.");

  vertex_t range_size{};
  if constexpr (GraphViewType::is_storage_transposed) {  // major
    range_size = graph_view.local_edge_partition_dst_range_size(size_t{0});
  } else {  // minor
    range_size = graph_view.local_edge_partition_dst_range_size();
  }
  auto expected_num_values = std::is_same_v<T, bool>
                               ? cugraph::packed_bool_size(static_cast<size_t>(range_size))
                               : static_cast<size_t>(range_size);
  CUGRAPH_EXPECTS(
    num_values == expected_num_values,
    "Invalid input argument: num_values does not match the expected number of values.");

  if constexpr (GraphViewType::is_storage_transposed) {  // major
    return detail::edge_endpoint_property_view_t<vertex_t, value_iterator, T>(
      std::vector<value_iterator>{value_first}, std::vector<vertex_t>{vertex_t{0}});
  } else {  // minor
    return detail::edge_endpoint_property_view_t<vertex_t, value_iterator, T>(value_first,
                                                                              vertex_t{0});
  }
}

template <typename vertex_t, typename... Iters, typename... Types>
auto view_concat(detail::edge_endpoint_property_view_t<vertex_t, Iters, Types> const&... views)
{
  auto first_view = get_first_of_pack(views...);
  if (first_view.is_major()) {
    return detail::major_view_concat(views...);
  } else {
    return detail::minor_view_concat(views...);
  }
}

}  // namespace cugraph

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cugraph/utilities/thrust_tuple_utils.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>

#include <optional>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename ValueIterator>
class rowcol_properties_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  rowcol_properties_device_view_t() = default;

  // SG & MG minor with value array
  rowcol_properties_device_view_t(ValueIterator value_first)
    : matrix_partition_value_first_(value_first)
  {
  }

  // MG major with value array
  rowcol_properties_device_view_t(ValueIterator value_first,
                                  vertex_t const* matrix_partition_major_value_start_offsets)
    : value_first_(value_first),
      matrix_partition_major_value_start_offsets_(matrix_partition_major_value_start_offsets)
  {
    set_local_adj_matrix_partition_idx(size_t{0});
  }

  // MG major with (k,v) pairs
  rowcol_properties_device_view_t(vertex_t const* key_first,
                                  vertex_t const* key_last,
                                  ValueIterator value_first,
                                  vertex_t const* matrix_partition_key_offsets,
                                  vertex_t const* matrix_partition_major_firsts)
    : key_first_(key_first),
      key_last_(key_last),
      value_first_(value_first),
      matrix_partition_key_offsets_(matrix_partition_key_offsets),
      matrix_partition_major_firsts_(matrix_partition_major_firsts)
  {
    set_local_adj_matrix_partition_idx(size_t{0});
  }

  // MG minor with (k,v) pairs
  rowcol_properties_device_view_t(vertex_t const* key_first,
                                  vertex_t const* key_last,
                                  ValueIterator value_first,
                                  vertex_t minor_first)
    : matrix_partition_key_first_(key_first),
      matrix_partition_key_last_(key_last),
      matrix_partition_id_first_(minor_first),
      matrix_partition_value_first_(value_first)
  {
  }

  void set_local_adj_matrix_partition_idx(size_t adj_matrix_partition_idx)
  {
    assert((adj_matrix_partition_idx == 0) || matrix_partition_major_value_start_offsets_ ||
           matrix_partition_key_offsets_);
    if (matrix_partition_major_value_start_offsets_) {
      matrix_partition_value_first_ =
        value_first_ + (*matrix_partition_major_value_start_offsets_)[adj_matrix_partition_idx];
    } else if (matrix_partition_key_offsets_) {
      matrix_partition_key_first_ =
        *key_first_ + (*matrix_partition_key_offsets_)[adj_matrix_partition_idx];
      matrix_partition_key_last_ =
        *key_first_ + (*matrix_partition_key_offsets_)[adj_matrix_partition_idx + 1];
      matrix_partition_id_first_ = (*matrix_partition_major_firsts_)[adj_matrix_partition_idx];
      matrix_partition_value_first_ =
        value_first_ + (*matrix_partition_key_offsets_)[adj_matrix_partition_idx];
    } else {
      assert(adj_matrix_partition_idx == 0);
    }
  }

  std::optional<vertex_t const*> key_first() const
  {
    return key_first_ ? std::optional<vertex_t const*>{*key_first_} : std::nullopt;
  }

  std::optional<vertex_t const*> key_last() const
  {
    return key_last_ ? std::optional<vertex_t const*>{*key_last_} : std::nullopt;
  }

  ValueIterator value_first() const { return value_first_; }

  std::optional<vertex_t const*> matrix_partition_major_value_start_offsets() const
  {
    return matrix_partition_major_value_start_offsets_
             ? std::optional<vertex_t const*>{*matrix_partition_major_value_start_offsets_}
             : std::nullopt;
  }

  std::optional<vertex_t const*> matrix_partition_key_offsets() const
  {
    return matrix_partition_key_offsets_
             ? std::optional<vertex_t const*>{*matrix_partition_key_offsets_}
             : std::nullopt;
  }

  std::optional<vertex_t const*> matrix_partition_major_firsts() const
  {
    return matrix_partition_major_firsts_
             ? std::optional<vertex_t const*>{*matrix_partition_major_firsts_}
             : std::nullopt;
  }

  std::optional<vertex_t const*> matrix_partition_key_first() const
  {
    return matrix_partition_key_first_
             ? std::optional<vertex_t const*>{*matrix_partition_key_first_}
             : std::nullopt;
  }

  std::optional<vertex_t const*> matrix_partition_key_last() const
  {
    return matrix_partition_key_last_ ? std::optional<vertex_t const*>{*matrix_partition_key_last_}
                                      : std::nullopt;
  }

  std::optional<vertex_t> matrix_partition_id_first() const
  {
    return matrix_partition_id_first_ ? std::optional<vertex_t>{*matrix_partition_id_first_}
                                      : std::nullopt;
  }

  ValueIterator matrix_partition_value_first() const { return matrix_partition_value_first_; }

  __device__ ValueIterator get_iter(vertex_t offset) const
  {
    auto value_offset = offset;
    if (matrix_partition_key_first_) {
      auto it = thrust::lower_bound(thrust::seq,
                                    *matrix_partition_key_first_,
                                    *matrix_partition_key_last_,
                                    *matrix_partition_id_first_ + offset);
      assert((it != *matrix_partition_key_last_) && (*it == (*matrix_partition_first_ + offset)));
      value_offset = static_cast<vertex_t>(thrust::distance(*matrix_partition_key_first_, it));
    }
    return matrix_partition_value_first_ + value_offset;
  }

  __device__ value_type get(vertex_t offset) const { return *get_iter(offset); }

 private:
  // MG major

  thrust::optional<vertex_t const*> key_first_{thrust::nullopt};
  thrust::optional<vertex_t const*> key_last_{thrust::nullopt};
  ValueIterator value_first_{};

  thrust::optional<vertex_t const*> matrix_partition_major_value_start_offsets_{
    thrust::nullopt};  // host data, for value array

  thrust::optional<vertex_t const*> matrix_partition_key_offsets_{
    thrust::nullopt};  // host data, for (k,v) pairs
  thrust::optional<vertex_t const*> matrix_partition_major_firsts_{
    thrust::nullopt};  // host data, for (k,v) pairs

  // for device functions (get_iter & get)

  thrust::optional<vertex_t const*> matrix_partition_key_first_{thrust::nullopt};
  thrust::optional<vertex_t const*> matrix_partition_key_last_{thrust::nullopt};
  thrust::optional<vertex_t> matrix_partition_id_first_{thrust::nullopt};

  ValueIterator matrix_partition_value_first_{};
};

template <typename vertex_t, typename T>
class rowcol_properties_t {
 public:
  rowcol_properties_t() : buffer_(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{})) {}

  // SG & MG minor with value array
  rowcol_properties_t(raft::handle_t const& handle, vertex_t buffer_size)
    : buffer_(allocate_dataframe_buffer<T>(buffer_size, handle.get_stream()))
  {
  }

  // MG major with value array
  rowcol_properties_t(raft::handle_t const& handle,
                      vertex_t buffer_size,
                      std::vector<vertex_t>&& matrix_partition_major_value_start_offsets)
    : buffer_(allocate_dataframe_buffer<T>(buffer_size, handle.get_stream())),
      matrix_partition_major_value_start_offsets_(
        std::move(matrix_partition_major_value_start_offsets))
  {
  }

  // MG major with (k,v) pairs
  rowcol_properties_t(raft::handle_t const& handle,
                      vertex_t const* key_first,
                      vertex_t const* key_last,
                      std::vector<vertex_t>&& matrix_partition_key_offsets,
                      std::vector<vertex_t>&& matrix_partition_major_firsts)
    : key_first_(key_first),
      key_last_(key_last),
      buffer_(
        allocate_dataframe_buffer<T>(matrix_partition_key_offsets.back(), handle.get_stream())),
      matrix_partition_key_offsets_(std::move(matrix_partition_key_offsets)),
      matrix_partition_major_firsts_(std::move(matrix_partition_major_firsts))
  {
  }

  // MG minor with (k,v) pairs
  rowcol_properties_t(raft::handle_t const& handle,
                      vertex_t const* key_first,
                      vertex_t const* key_last,
                      vertex_t minor_first)
    : key_first_(key_first),
      key_last_(key_last),
      buffer_(
        allocate_dataframe_buffer<T>(thrust::distance(key_first, key_last), handle.get_stream())),
      minor_first_(minor_first)
  {
  }

  void fill(T value, rmm::cuda_stream_view stream)
  {
    thrust::fill(rmm::exec_policy(stream),
                 value_first(),
                 value_first() + size_dataframe_buffer(buffer_),
                 value);
  }

  auto key_first() { return key_first_; }
  auto key_last() { return key_last_; }
  auto value_first() { return get_dataframe_buffer_begin(buffer_); }

  auto device_view() const
  {
    auto value_first = get_dataframe_buffer_cbegin(buffer_);
    if (matrix_partition_major_value_start_offsets_) {  // MG major with value array
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(
        value_first, (*matrix_partition_major_value_start_offsets_).data());
    } else if (matrix_partition_key_offsets_) {  // MG major with (k,v) pairs
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(
        *key_first_,
        *key_last_,
        value_first,
        (*matrix_partition_key_offsets_).data(),
        (*matrix_partition_major_firsts_).data());
    } else if (minor_first_) {  // MG minor with (k,v) pairs
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(
        *key_first_, *key_last_, value_first, *minor_first_);
    } else {  // SG & MG minor with value array
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(value_first);
    }
  }

  auto mutable_device_view()
  {
    auto value_first = get_dataframe_buffer_begin(buffer_);
    if (matrix_partition_major_value_start_offsets_) {  // MG major with value array
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(
        value_first, (*matrix_partition_major_value_start_offsets_).data());
    } else if (matrix_partition_key_offsets_) {  // MG major with (k,v) pairs
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(
        *key_first_,
        *key_last_,
        value_first,
        (*matrix_partition_key_offsets_).data(),
        (*matrix_partition_major_firsts_).data());
    } else if (minor_first_) {  // MG minor with (k,v) pairs
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(
        *key_first_, *key_last_, value_first, *minor_first_);
    } else {  // SG & MG minor with value array
      return rowcol_properties_device_view_t<vertex_t, decltype(value_first)>(value_first);
    }
  }

 private:
  std::optional<vertex_t const*> key_first_{std::nullopt};
  std::optional<vertex_t const*> key_last_{std::nullopt};

  decltype(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{})) buffer_;

  // MG major

  std::optional<std::vector<vertex_t>> matrix_partition_major_value_start_offsets_{
    std::nullopt};  // for value array

  std::optional<std::vector<vertex_t>> matrix_partition_key_offsets_{
    std::nullopt};  // for (k,v) pairs
  std::optional<std::vector<vertex_t>> matrix_partition_major_firsts_{
    std::nullopt};  // for (k,v) pairs

  // MG minor

  std::optional<vertex_t> minor_first_{std::nullopt};  // for (k,v) pairs
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
class row_properties_t {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  row_properties_t() = default;

  row_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    auto key_first = graph_view.get_local_sorted_unique_edge_row_begin();
    if (key_first) {
      if constexpr (GraphViewType::is_multi_gpu) {
        auto key_last = graph_view.get_local_sorted_unique_edge_row_end();
        if (graph_view.storage_transposed()) {  // minor
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle, *key_first, *key_last, graph_view.get_local_adj_matrix_partition_row_first());
        } else {  // major
          std::vector<vertex_t> matrix_partition_major_firsts(
            graph_view.get_number_of_local_adj_matrix_partitions());
          for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
            matrix_partition_major_firsts[i] =
              graph_view.get_local_adj_matrix_partition_row_first(i);
          }
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle,
            *key_first,
            *key_last,
            *(graph_view.get_local_sorted_unique_edge_row_offsets()),
            std::move(matrix_partition_major_firsts));
        }
      } else {
        assert(false);
      }
    } else {
      if constexpr (GraphViewType::is_multi_gpu) {
        if (graph_view.storage_transposed()) {  // minor
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle, graph_view.get_number_of_local_adj_matrix_partition_rows());
        } else {  // major
          std::vector<vertex_t> matrix_partition_major_value_start_offsets(
            graph_view.get_number_of_local_adj_matrix_partitions());
          for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
            matrix_partition_major_value_start_offsets[i] =
              graph_view.get_local_adj_matrix_partition_row_value_start_offset(i);
          }
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle,
            graph_view.get_number_of_local_adj_matrix_partition_rows(),
            std::move(matrix_partition_major_value_start_offsets));
        }
      } else {
        properties_ = detail::rowcol_properties_t<vertex_t, T>(
          handle, graph_view.get_number_of_local_adj_matrix_partition_rows());
      }
    }
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto key_first() { return properties_.key_first(); }
  auto key_last() { return properties_.key_last(); }

  auto value_first() { return properties_.value_first(); }

  auto device_view() const { return properties_.device_view(); }
  auto mutable_device_view() { return properties_.mutable_device_view(); }

 private:
  detail::rowcol_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

template <typename GraphViewType, typename T>
class col_properties_t {
 public:
  using value_type = T;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  col_properties_t() = default;

  col_properties_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    auto key_first = graph_view.get_local_sorted_unique_edge_col_begin();
    if (key_first) {
      if constexpr (GraphViewType::is_multi_gpu) {
        auto key_last = graph_view.get_local_sorted_unique_edge_col_end();
        if (graph_view.storage_transposed()) {  // major
          std::vector<vertex_t> matrix_partition_major_firsts(
            graph_view.get_number_of_local_adj_matrix_partitions());
          for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
            matrix_partition_major_firsts[i] =
              graph_view.get_local_adj_matrix_partition_col_first(i);
          }
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle,
            *key_first,
            *key_last,
            *(graph_view.get_local_sorted_unique_edge_col_offsets()),
            std::move(matrix_partition_major_firsts));
        } else {  // minor
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle, *key_first, *key_last, graph_view.get_local_adj_matrix_partition_col_first());
        }
      } else {
        assert(false);
      }
    } else {
      if constexpr (GraphViewType::is_multi_gpu) {
        if (graph_view.storage_transposed()) {  // major
          std::vector<vertex_t> matrix_partition_major_value_start_offsets(
            graph_view.get_number_of_local_adj_matrix_partitions());
          for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
            matrix_partition_major_value_start_offsets[i] =
              graph_view.get_local_adj_matrix_partition_col_value_start_offset(i);
          }
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle,
            graph_view.get_number_of_local_adj_matrix_partition_cols(),
            std::move(matrix_partition_major_value_start_offsets));
        } else {  // minor
          properties_ = detail::rowcol_properties_t<vertex_t, T>(
            handle, graph_view.get_number_of_local_adj_matrix_partition_cols());
        }
      } else {
        properties_ = detail::rowcol_properties_t<vertex_t, T>(
          handle, graph_view.get_number_of_local_adj_matrix_partition_cols());
      }
    }
  }

  void fill(T value, rmm::cuda_stream_view stream) { properties_.fill(value, stream); }

  auto key_first() { return properties_.key_first(); }
  auto key_last() { return properties_.key_last(); }

  auto value_first() { return properties_.value_first(); }

  auto device_view() const { return properties_.device_view(); }
  auto mutable_device_view() { return properties_.mutable_device_view(); }

 private:
  detail::rowcol_properties_t<typename GraphViewType::vertex_type, T> properties_{};
};

template <typename vertex_t>
class dummy_properties_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  void set_local_adj_matrix_partition_idx(size_t adj_matrix_partition_idx) {}  // no-op

  __device__ auto get(vertex_t offset) const { return thrust::nullopt; }
};

template <typename vertex_t>
class dummy_properties_t {
 public:
  using value_type = thrust::nullopt_t;

  auto device_view() const { return dummy_properties_device_view_t<vertex_t>{}; }
};

template <typename vertex_t, typename... Ts>
auto device_view_concat(
  detail::rowcol_properties_device_view_t<vertex_t, Ts> const&... device_views)
{
  auto concat_first = thrust::make_zip_iterator(
    thrust_tuple_cat(detail::to_thrust_tuple(device_views.value_first())...));
  auto first = detail::get_first_of_pack(device_views...);
  if (first.matrix_partition_major_value_start_offsets()) {  // MG major with value array
    return detail::rowcol_properties_device_view_t<vertex_t, decltype(concat_first)>(
      concat_first, *(first.matrix_partition_major_value_start_offsets()));
  } else if (first.matrix_partition_key_offsets()) {  // MG major with (k,v) pairs
    return detail::rowcol_properties_device_view_t<vertex_t, decltype(concat_first)>(
      *(first.key_first()),
      *(first.key_last()),
      concat_first,
      *(first.matrix_partition_key_offsets()),
      *(first.matrix_partition_major_firsts()));
  } else if (first.matrix_partition_key_first()) {  // MG minor with (k,v) pairs
    return detail::rowcol_properties_device_view_t<vertex_t, decltype(concat_first)>(
      *(first.matrix_partition_key_first()),
      *(first.matrix_partition_key_last()),
      concat_first,
      *(first.matrix_partition_id_first()));
  } else {  // SG & MG minor with value array
    return detail::rowcol_properties_device_view_t<vertex_t, decltype(concat_first)>(concat_first);
  }
}

}  // namespace cugraph

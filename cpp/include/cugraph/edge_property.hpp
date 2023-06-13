/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

#include <optional>
#include <type_traits>

namespace cugraph {

template <typename edge_t,
          typename ValueIterator,
          typename value_t = typename thrust::iterator_traits<ValueIterator>::value_type>
class edge_property_view_t {
 public:
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<ValueIterator>::value_type, value_t> ||
    cugraph::has_packed_bool_element<ValueIterator, value_t>());

  using edge_type      = edge_t;
  using value_type     = value_t;
  using value_iterator = ValueIterator;

  edge_property_view_t() = default;

  edge_property_view_t(std::vector<ValueIterator> const& edge_partition_value_firsts,
                       std::vector<edge_t> const& edge_partition_edge_counts)
    : edge_partition_value_firsts_(edge_partition_value_firsts),
      edge_partition_edge_counts_(edge_partition_edge_counts)
  {
  }

  std::vector<ValueIterator> const& value_firsts() const { return edge_partition_value_firsts_; }

  std::vector<edge_t> const& edge_counts() const { return edge_partition_edge_counts_; }

 private:
  std::vector<ValueIterator> edge_partition_value_firsts_{};
  std::vector<edge_t> edge_partition_edge_counts_{};
};

class edge_dummy_property_view_t {
 public:
  using value_type     = thrust::nullopt_t;
  using value_iterator = void*;
};

template <typename GraphViewType, typename T>
class edge_property_t {
 public:
  static_assert(cugraph::is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using edge_type   = typename GraphViewType::edge_type;
  using value_type  = T;
  using buffer_type = decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}));

  edge_property_t(raft::handle_t const& handle) {}

  edge_property_t(raft::handle_t const& handle, GraphViewType const& graph_view)
  {
    buffers_.reserve(graph_view.number_of_local_edge_partitions());
    edge_counts_ = std::vector<edge_type>(graph_view.number_of_local_edge_partitions(), 0);
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto num_edges =
        static_cast<size_t>(graph_view.local_edge_partition_view(i).number_of_edges());
      size_t buffer_size =
        std::is_same_v<T, bool> ? cugraph::packed_bool_size(num_edges) : num_edges;
      buffers_.push_back(
        allocate_dataframe_buffer<std::conditional_t<std::is_same_v<T, bool>, uint32_t, T>>(
          buffer_size, handle.get_stream()));
      edge_counts_[i] = num_edges;
    }
  }

  template <typename value_type = T, typename = std::enable_if_t<!std::is_same_v<value_type, bool>>>
  edge_property_t(std::vector<buffer_type>&& buffers) : buffers_(std::move(buffers))
  {
    edge_counts_.resize(buffers_.size());
    for (size_t i = 0; i < edge_counts_.size(); ++i) {
      edge_counts_[i] = size_dataframe_buffer(buffers_[i]);
    }
  }

  edge_property_t(std::vector<buffer_type>&& buffers, std::vector<edge_type>&& edge_counts)
    : buffers_(std::move(buffers)), edge_counts_(std::move(edge_counts))
  {
  }

  void clear(raft::handle_t const& handle)
  {
    buffers_.clear();
    buffers_.shrink_to_fit();
    edge_counts_.clear();
    edge_counts_.shrink_to_fit();
  }

  auto view() const
  {
    using const_value_iterator = decltype(get_dataframe_buffer_cbegin(buffers_[0]));

    std::vector<const_value_iterator> edge_partition_value_firsts(buffers_.size());
    std::vector<edge_type> edge_partition_edge_counts(buffers_.size());
    for (size_t i = 0; i < edge_partition_value_firsts.size(); ++i) {
      edge_partition_value_firsts[i] = get_dataframe_buffer_cbegin(buffers_[i]);
      edge_partition_edge_counts[i]  = edge_counts_[i];
    }

    return edge_property_view_t<edge_type, const_value_iterator, T>(edge_partition_value_firsts,
                                                                    edge_partition_edge_counts);
  }

  auto mutable_view()
  {
    using value_iterator = decltype(get_dataframe_buffer_begin(buffers_[0]));

    std::vector<value_iterator> edge_partition_value_firsts(buffers_.size());
    std::vector<edge_type> edge_partition_edge_counts(buffers_.size());
    for (size_t i = 0; i < edge_partition_value_firsts.size(); ++i) {
      edge_partition_value_firsts[i] = get_dataframe_buffer_begin(buffers_[i]);
      edge_partition_edge_counts[i]  = edge_counts_[i];
    }

    return edge_property_view_t<edge_type, value_iterator, T>(edge_partition_value_firsts,
                                                              edge_partition_edge_counts);
  }

 private:
  std::vector<buffer_type> buffers_{};
  std::vector<edge_type> edge_counts_{};
};

class edge_dummy_property_t {
 public:
  using value_type = thrust::nullopt_t;

  auto view() const { return edge_dummy_property_view_t{}; }
};

template <typename edge_t, typename... Iters, typename... Types>
auto view_concat(edge_property_view_t<edge_t, Iters, Types> const&... views)
{
  using concat_value_iterator = decltype(thrust::make_zip_iterator(
    thrust_tuple_cat(to_thrust_iterator_tuple(views.value_firsts()[0])...)));
  using concat_value_type     = decltype(thrust_tuple_cat(to_thrust_tuple(Types{})...));

  std::vector<concat_value_iterator> edge_partition_concat_value_firsts{};
  auto first_view = get_first_of_pack(views...);
  edge_partition_concat_value_firsts.resize(first_view.value_firsts().size());
  for (size_t i = 0; i < edge_partition_concat_value_firsts.size(); ++i) {
    edge_partition_concat_value_firsts[i] = thrust::make_zip_iterator(
      thrust_tuple_cat(to_thrust_iterator_tuple(views.value_firsts()[i])...));
  }

  return edge_property_view_t<edge_t, concat_value_iterator, concat_value_type>(
    edge_partition_concat_value_firsts, first_view.edge_counts());
}

}  // namespace cugraph

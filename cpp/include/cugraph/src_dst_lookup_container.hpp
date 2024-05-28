/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/tuple.h>

#include <unordered_map>
#include <vector>

namespace cugraph {

namespace detail {

template <typename TupleType, std::size_t... Is>
constexpr TupleType invalid_of_thrust_tuple_of_integral(std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    cugraph::invalid_idx<typename thrust::tuple_element<Is, TupleType>::type>::value...);
}
}  // namespace detail

template <typename TupleType>
constexpr TupleType invalid_of_thrust_tuple_of_integral()
{
  return detail::invalid_of_thrust_tuple_of_integral<TupleType>(
    std::make_index_sequence<thrust::tuple_size<TupleType>::value>());
}

template <typename edge_id_t, typename edge_type_t, typename value_t>
class search_container_t {
  template <typename _edge_id_t, typename _edge_type_t, typename _value_t>
  struct impl;
  std::unique_ptr<impl<edge_id_t, edge_type_t, value_t>> pimpl;

 public:
  using edge_id_type   = edge_id_t;
  using edge_type_type = edge_type_t;
  using value_type     = value_t;

  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<value_t>::value);

  ~search_container_t();
  search_container_t();
  search_container_t(raft::handle_t const& handle,
                     std::vector<edge_type_t> types,
                     std::vector<edge_id_t> type_counts);
  search_container_t(const search_container_t&);

  void insert(raft::handle_t const& handle,
              edge_type_t typ,
              raft::device_span<edge_id_t const> edge_ids_to_insert,
              decltype(cugraph::allocate_dataframe_buffer<value_t>(
                0, rmm::cuda_stream_view{}))&& values_to_insert);

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  lookup_src_dst_from_edge_id_and_type(raft::handle_t const& handle,
                                       raft::device_span<edge_id_t const> edge_ids_to_lookup,
                                       edge_type_t edge_type_to_lookup,
                                       bool multi_gpu) const;

  std::optional<decltype(cugraph::allocate_dataframe_buffer<value_t>(0, rmm::cuda_stream_view{}))>
  lookup_src_dst_from_edge_id_and_type(raft::handle_t const& handle,
                                       raft::device_span<edge_id_t const> edge_ids_to_lookup,
                                       raft::device_span<edge_type_t const> edge_types_to_lookup,
                                       bool multi_gpu) const;
  void print() const;
};

}  // namespace cugraph

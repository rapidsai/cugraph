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

template <typename edge_id_t,
          typename edge_type_t,
          typename vertex_t,
          typename value_t = thrust::tuple<vertex_t, vertex_t>>
class lookup_container_t {
  template <typename _edge_id_t, typename _edge_type_t, typename _vertex_t, typename _value_t>
  struct lookup_container_impl;
  std::unique_ptr<lookup_container_impl<edge_id_t, edge_type_t, vertex_t, value_t>> pimpl;

 public:
  using edge_id_type   = edge_id_t;
  using edge_type_type = edge_type_t;
  using value_type     = value_t;

  static_assert(std::is_integral_v<edge_id_t>);
  static_assert(std::is_integral_v<edge_type_t>);
  static_assert(is_thrust_tuple_of_integral<value_t>::value);

  ~lookup_container_t();
  lookup_container_t();
  lookup_container_t(raft::handle_t const& handle,
                     std::vector<edge_type_t> types,
                     std::vector<edge_id_t> type_counts);
  lookup_container_t(const lookup_container_t&);

  void insert(raft::handle_t const& handle,
              edge_type_t typ,
              raft::device_span<edge_id_t const> edge_ids_to_insert,
              dataframe_buffer_type_t<value_t>&& values_to_insert);

  dataframe_buffer_type_t<value_t> src_dst_from_edge_id_and_type(
    raft::handle_t const& handle,
    raft::device_span<edge_id_t const> edge_ids_to_lookup,
    edge_type_t edge_type_to_lookup,
    bool multi_gpu) const;

  dataframe_buffer_type_t<value_t> src_dst_from_edge_id_and_type(
    raft::handle_t const& handle,
    raft::device_span<edge_id_t const> edge_ids_to_lookup,
    raft::device_span<edge_type_t const> edge_types_to_lookup,
    bool multi_gpu) const;
};

}  // namespace cugraph

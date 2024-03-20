/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>
#include <vector>

namespace cugraph {
namespace test {

template <typename GraphViewType, typename property_t>
struct generate {
 private:
  using vertex_type = typename GraphViewType::vertex_type;

  using property_buffer_type = std::decay_t<decltype(allocate_dataframe_buffer<property_t>(
    size_t{0}, rmm::cuda_stream_view{}))>;

 public:
  static property_t initial_value(int32_t init);

  static property_buffer_type vertex_property(raft::handle_t const& handle,
                                              rmm::device_uvector<vertex_type> const& labels,
                                              int32_t hash_bin_count);

  static property_buffer_type vertex_property(raft::handle_t const& handle,
                                              thrust::counting_iterator<vertex_type> begin,
                                              thrust::counting_iterator<vertex_type> end,
                                              int32_t hash_bin_count);

  static cugraph::edge_src_property_t<GraphViewType, property_t> src_property(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    property_buffer_type const& property);

  static cugraph::edge_dst_property_t<GraphViewType, property_t> dst_property(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    property_buffer_type const& property);

  static cugraph::edge_property_t<GraphViewType, property_t> edge_property(
    raft::handle_t const& handle, GraphViewType const& graph_view, int32_t hash_bin_count);
};

}  // namespace test
}  // namespace cugraph

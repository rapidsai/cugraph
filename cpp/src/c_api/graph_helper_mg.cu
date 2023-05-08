/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <c_api/graph_helper_impl.cuh>

namespace cugraph {
namespace c_api {

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, true>, float>
create_constant_edge_property(raft::handle_t const& handle,
                              cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                              float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, true>, float>
create_constant_edge_property(raft::handle_t const& handle,
                              cugraph::graph_view_t<int32_t, int64_t, true, true> const& graph_view,
                              float constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, true>, float>
create_constant_edge_property(raft::handle_t const& handle,
                              cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                              float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, true>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, true>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, true>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, true>, double>
create_constant_edge_property(raft::handle_t const& handle,
                              cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
                              double constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, true>, double>
create_constant_edge_property(raft::handle_t const& handle,
                              cugraph::graph_view_t<int32_t, int64_t, true, true> const& graph_view,
                              double constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, true>, double>
create_constant_edge_property(raft::handle_t const& handle,
                              cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
                              double constant_value);

}  // namespace c_api
}  // namespace cugraph

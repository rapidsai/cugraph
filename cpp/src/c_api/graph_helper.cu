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

#include <c_api/graph_helper_impl.cuh>

namespace cugraph {
namespace c_api {

template rmm::device_uvector<int32_t> expand_sparse_offsets(
  raft::device_span<int32_t const> offsets,
  int32_t base_vertex_id,
  rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int32_t> expand_sparse_offsets(
  raft::device_span<int64_t const> offsets,
  int32_t base_vertex_id,
  rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int64_t> expand_sparse_offsets(
  raft::device_span<int64_t const> offsets,
  int64_t base_vertex_id,
  rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int32_t> expand_sparse_offsets(raft::device_span<size_t const> offsets,
                                                            int32_t base_vertex_id,
                                                            rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int64_t> expand_sparse_offsets(raft::device_span<size_t const> offsets,
                                                            int64_t base_vertex_id,
                                                            rmm::cuda_stream_view const& stream);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, false>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, false>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, false>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, false>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, false>, float>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  float constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, false>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, false>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, false>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, false>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  double constant_value);

template edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, false>, double>
create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  double constant_value);

}  // namespace c_api
}  // namespace cugraph

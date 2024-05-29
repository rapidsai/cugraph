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

#include "lookup/lookup_src_dst_impl.cuh"

namespace cugraph {

template class search_container_t<int32_t, int32_t, thrust::tuple<int32_t, int32_t>>;
template class search_container_t<int64_t, int32_t, thrust::tuple<int32_t, int32_t>>;
template class search_container_t<int64_t, int32_t, thrust::tuple<int64_t, int64_t>>;

template search_container_t<int32_t, int32_t, thrust::tuple<int32_t, int32_t>>
build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, int32_t const*> edge_id_view,
  edge_property_view_t<int32_t, int32_t const*> edge_type_view);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
lookup_src_dst_from_edge_id_and_type<int32_t, int32_t, int32_t, true>(
  raft::handle_t const& handle,
  search_container_t<int32_t, int32_t, thrust::tuple<int32_t, int32_t>> const& search_container,
  raft::device_span<int32_t const> edge_ids_to_lookup,
  int32_t edge_type_to_lookup);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
lookup_src_dst_from_edge_id_and_type<int32_t, int32_t, int32_t, true>(
  raft::handle_t const& handle,
  search_container_t<int32_t, int32_t, thrust::tuple<int32_t, int32_t>> const& search_container,
  raft::device_span<int32_t const> edge_ids_to_lookup,
  raft::device_span<int32_t const> edge_types_to_lookup);

template search_container_t<int64_t, int32_t, thrust::tuple<int32_t, int32_t>>
build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, int64_t const*> edge_id_view,
  edge_property_view_t<int64_t, int32_t const*> edge_type_view);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
lookup_src_dst_from_edge_id_and_type<int32_t, int64_t, int32_t, true>(
  raft::handle_t const& handle,
  search_container_t<int64_t, int32_t, thrust::tuple<int32_t, int32_t>> const& search_container,
  raft::device_span<int64_t const> edge_ids_to_lookup,
  int32_t edge_type_to_lookup);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
lookup_src_dst_from_edge_id_and_type<int32_t, int64_t, int32_t, true>(
  raft::handle_t const& handle,
  search_container_t<int64_t, int32_t, thrust::tuple<int32_t, int32_t>> const& search_container,
  raft::device_span<int64_t const> edge_ids_to_lookup,
  raft::device_span<int32_t const> edge_types_to_lookup);

template search_container_t<int64_t, int32_t, thrust::tuple<int64_t, int64_t>>
build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, int64_t const*> edge_id_view,
  edge_property_view_t<int64_t, int32_t const*> edge_type_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
lookup_src_dst_from_edge_id_and_type<int64_t, int64_t, int32_t, true>(
  raft::handle_t const& handle,
  search_container_t<int64_t, int32_t, thrust::tuple<int64_t, int64_t>> const& search_container,
  raft::device_span<int64_t const> edge_ids_to_lookup,
  int32_t edge_type_to_lookup);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
lookup_src_dst_from_edge_id_and_type<int64_t, int64_t, int32_t, true>(
  raft::handle_t const& handle,
  search_container_t<int64_t, int32_t, thrust::tuple<int64_t, int64_t>> const& search_container,
  raft::device_span<int64_t const> edge_ids_to_lookup,
  raft::device_span<int32_t const> edge_types_to_lookup);

}  // namespace cugraph

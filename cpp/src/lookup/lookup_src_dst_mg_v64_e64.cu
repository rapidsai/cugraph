/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lookup/lookup_src_dst_impl.cuh"

namespace cugraph {

template class lookup_container_t<int64_t, int32_t, int64_t>;

template lookup_container_t<int64_t, int32_t, int64_t> build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, int64_t const*> edge_id_view,
  edge_property_view_t<int64_t, int32_t const*> edge_type_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
lookup_endpoints_from_edge_ids_and_single_type<int64_t, int64_t, int32_t, true>(
  raft::handle_t const& handle,
  lookup_container_t<int64_t, int32_t, int64_t> const& search_container,
  raft::device_span<int64_t const> edge_ids_to_lookup,
  int32_t edge_type_to_lookup);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
lookup_endpoints_from_edge_ids_and_types<int64_t, int64_t, int32_t, true>(
  raft::handle_t const& handle,
  lookup_container_t<int64_t, int32_t, int64_t> const& search_container,
  raft::device_span<int64_t const> edge_ids_to_lookup,
  raft::device_span<int32_t const> edge_types_to_lookup);

}  // namespace cugraph

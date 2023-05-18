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

#include <prims/fill_edge_property.cuh>
#include <structure/detail/structure_utils.cuh>

namespace cugraph {
namespace c_api {

template <typename vertex_t, typename edge_t>
rmm::device_uvector<vertex_t> expand_sparse_offsets(raft::device_span<edge_t const> offsets,
                                                    vertex_t base_vertex_id,
                                                    rmm::cuda_stream_view const& stream)
{
  return cugraph::detail::expand_sparse_offsets(offsets, base_vertex_id, stream);
}

template <typename GraphViewType, typename T>
edge_property_t<GraphViewType, T> create_constant_edge_property(raft::handle_t const& handle,
                                                                GraphViewType const& graph_view,
                                                                T constant_value)
{
  edge_property_t<GraphViewType, T> edge_property(handle, graph_view);

  cugraph::fill_edge_property(handle, graph_view, constant_value, edge_property);

  return edge_property;
}

}  // namespace c_api
}  // namespace cugraph

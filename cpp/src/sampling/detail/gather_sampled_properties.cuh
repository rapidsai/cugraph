/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "prims/edge_bucket.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

namespace cugraph {
namespace detail {

/**
 * @brief Gather properties
 *
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::vector<arithmetic_device_uvector_t> gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::edge_bucket_t<vertex_t, edge_t, true, multi_gpu, false>& edge_list,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views);

}  // namespace detail
}  // namespace cugraph

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
#pragma once

#include <structure/detail/structure_utils.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <optional>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>>
remove_self_loops(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edgelist_srcs,
                  rmm::device_uvector<vertex_t>&& edgelist_dsts,
                  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                  std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types)
{
  auto [remove_count, remove_flags] = detail::mark_edges_for_removal(
    handle,
    raft::device_span<vertex_t const>{edgelist_srcs.data(), edgelist_srcs.size()},
    raft::device_span<vertex_t const>{edgelist_dsts.data(), edgelist_dsts.size()},
    [d_srcs = edgelist_srcs.data(), d_dsts = edgelist_dsts.data()] __device__(size_t i) {
      return d_srcs[i] == d_dsts[i];
    });

  if (remove_count > 0) {
    edgelist_srcs =
      detail::remove_flagged_elements(handle, std::move(edgelist_srcs), remove_flags, remove_count);
    edgelist_dsts =
      detail::remove_flagged_elements(handle, std::move(edgelist_dsts), remove_flags, remove_count);

    if (edgelist_weights)
      edgelist_weights = detail::remove_flagged_elements(
        handle, std::move(*edgelist_weights), remove_flags, remove_count);

    if (edgelist_edge_ids)
      edgelist_edge_ids = detail::remove_flagged_elements(
        handle, std::move(*edgelist_edge_ids), remove_flags, remove_count);

    if (edgelist_edge_types)
      edgelist_edge_types = detail::remove_flagged_elements(
        handle, std::move(*edgelist_edge_types), remove_flags, remove_count);
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types));
}

}  // namespace cugraph
